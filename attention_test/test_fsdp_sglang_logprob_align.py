#!/usr/bin/env python3
"""
=============================================================================
FSDP vs SGLang Logprob 对齐测试（SGLang 存 → FSDP 读 → teacher-forcing）
=============================================================================
功能：验证 SGLang 推理与 FSDP-style forward 的 logprob 是否一致。
      用于 Miles true on-policy 场景：SGLang 做 rollout，FSDP 做训练，
      两者在相同 trajectory 上的 logprob 应对齐。

流程（SGLang 存 → FSDP 读 → teacher-forcing forward → 比对）：
  1. SGLang 贪婪生成 N 个 token，return_logprob=True，保存 tokens + rollout_logprobs 到 txt
  2. FSDP 侧读取同一批数据，对 tokens 做 teacher-forcing forward
  3. 比较 rollout_logprobs 与 FSDP forward 的 logprob

HF 侧可配置：
  - --attn-implementation: sdpa|eager|flash_attention_2|flash_attention_3（HF 无 triton）

用法：
  # 1. 先启动 SGLang（建议设置 SGLANG_RETURN_ORIGINAL_LOGPROB=1）
  #    FA3 对齐（NVIDIA，与 Miles true-on-policy 一致，需加 --rl-on-policy-target fsdp）：
  SGLANG_RETURN_ORIGINAL_LOGPROB=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \\
      --model-path Qwen/Qwen3-8B \\
      --attention-backend fa3 \\
      --mem-fraction-static 0.7 \\
      --host 0.0.0.0 --port 30000 \\
      --enable-deterministic-inference \\
      --disable-radix-cache \\
      --rl-on-policy-target fsdp

  #    Triton 对齐（无 FA3 时）：
  #  SGLANG_RETURN_ORIGINAL_LOGPROB=1 ... --attention-backend triton （不加 rl-on-policy-target）

  # 2. 运行本脚本
  python3 test_fsdp_sglang_logprob_align.py --host localhost --port 30000 --attn-implementation flash_attention_3
  python3 test_fsdp_sglang_logprob_align.py --host localhost --port 30000 --attn-implementation sdpa
  python3 test_fsdp_sglang_logprob_align.py --host localhost --port 30000 --attn-implementation eager
  python3 test_fsdp_sglang_logprob_align.py --max-new-tokens 20 --tolerance 1e-4

  # 两阶段模式
  python3 test_fsdp_sglang_logprob_align.py --host localhost --port 30000 --save-rollout results/rollout.txt
  python3 test_fsdp_sglang_logprob_align.py --load-rollout results/rollout.txt --attn-implementation sdpa

  # FSDP 侧使用 batch_invariant_ops（与 Miles 一致，enable_bmm=False）
  python3 test_fsdp_sglang_logprob_align.py --host localhost --port 30000 \\
      --attn-implementation flash_attention_3 --use-batch-invariant

  # 可选：提高确定性（与 Miles 一致）
  # export NCCL_ALGO=allreduce:tree NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8

输出：
  逐 token 对比报告，以及 PASS/FAIL 结论。

显存说明（8 卡 ROCm 等）：
  - 两阶段模式（--save-rollout / --load-rollout）：SGLang 与 HF 不同时驻留，单卡即可。
  - 一阶段模式：SGLang 服务占用 1 卡，脚本的 HF 模型占 1 卡，可指定不同 GPU。
  - Qwen3-8B 约 16GB（bf16），单卡 24GB+ 可运行。
=============================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

# 确保 /app/program 在 PYTHONPATH，以便 import batch_invariant_trace
_program_dir = os.path.dirname(os.path.abspath(__file__))
if _program_dir not in sys.path:
    sys.path.insert(0, _program_dir)

import requests
import torch


# =============================================================================
# 1. 参数解析
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="FSDP vs SGLang logprob 对齐测试（HF attn_implementation 可配置）"
    )
    parser.add_argument("--host", type=str, default="localhost", help="SGLang 服务器地址")
    parser.add_argument("--port", type=int, default=30000, help="SGLang 服务器端口")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="模型路径（需与 SGLang server 一致，用于 HF forward）",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3", "triton"],
        help="HF attn_implementation: sdpa|eager|flash_attention_2|flash_attention_3|triton",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请简单介绍下 SGLang，它是什么、能做什么。",
        help="测试用 prompt 文本",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="生成 token 数量",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="logprob 数值比较的容差（绝对值差）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="HF 模型加载设备",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="结果保存为 JSON 文件路径",
    )
    parser.add_argument(
        "--save-detail",
        type=str,
        default=None,
        help="保存 SGLang 与 HF 逐 token 人工对比详情（文本格式，便于学习）",
    )
    parser.add_argument(
        "--save-rollout",
        type=str,
        default=None,
        help="保存 SGLang rollout 到 txt（tokens、rollout_logprobs 等），供 --load-rollout 读取",
    )
    parser.add_argument(
        "--load-rollout",
        type=str,
        default=None,
        help="从 txt 加载 rollout 数据，跳过 SGLang，直接做 FSDP forward 并比对",
    )
    parser.add_argument(
        "--use-batch-invariant",
        action="store_true",
        help="FSDP 侧使用 batch_invariant_ops 替换 mm/addmm/bmm/log_softmax/mean（需安装 sglang）",
    )
    return parser.parse_args()


# =============================================================================
# 2. SGLang 生成 + 取 logprob
# =============================================================================


def sglang_generate_with_logprobs(
    host: str,
    port: int,
    prompt_ids: List[int],
    max_new_tokens: int,
) -> Tuple[List[int], List[float], List[int], List[str]]:
    """
    调用 SGLang /generate，贪婪生成，返回完整序列和 logprobs。

    Returns:
        full_token_ids: prompt_ids + gen_ids
        gen_logprobs: 每个生成 token 的 logprob（与 gen_ids 一一对应）
        gen_token_ids: 生成的 token ids
        gen_token_texts: 每个生成 token 的文本（用于人工对比）
    """
    base_url = f"http://{host}:{port}"
    json_data = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0.0,  # greedy
            "max_new_tokens": max_new_tokens,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "sampling_seed": 42,
        },
        "return_logprob": True,
        "return_text_in_logprobs": True,  # 返回 token 解码文本，否则 token_text 为空
        "stream": False,
    }

    response = requests.post(f"{base_url}/generate", json=json_data)
    if response.status_code != 200:
        raise RuntimeError(f"SGLang 请求失败: {response.status_code} {response.text}")

    ret = response.json()
    # 单请求时返回 [resp]
    if isinstance(ret, list):
        ret = ret[0]

    output_ids = ret["output_ids"]
    meta = ret.get("meta_info", {})
    output_token_logprobs = meta.get("output_token_logprobs", [])

    if not output_token_logprobs:
        raise RuntimeError(
            "SGLang 未返回 output_token_logprobs，请确认请求中 return_logprob=True"
        )

    # output_token_logprobs: [(logprob, token_id, text), ...]，与 output_ids 顺序一致
    gen_token_ids = output_ids
    n_gen = len(output_ids)
    gen_logprobs = [item[0] for item in output_token_logprobs[:n_gen]]
    gen_token_texts = [
        item[2] if len(item) > 2 else "" for item in output_token_logprobs[:n_gen]
    ]
    if len(gen_logprobs) < n_gen:
        raise RuntimeError(
            f"output_token_logprobs 长度 {len(gen_logprobs)} < output_ids {n_gen}"
        )

    full_token_ids = prompt_ids + gen_token_ids
    return full_token_ids, gen_logprobs, gen_token_ids, gen_token_texts


# =============================================================================
# 3. HuggingFace forward 模拟 Miles FSDP get_logprobs
# =============================================================================


def _compute_log_probs_miles_style(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """
    使用 Miles ppo_utils._calculate_log_probs_and_entropy_true_on_policy 计算 logprob（必须安装 miles）。
    """
    import sys, os
    _miles_dir = "/app/program/miles"
    if _miles_dir not in sys.path:
        sys.path.insert(0, _miles_dir)
    import miles.utils.ppo_utils as miles_ppo

    print(f"[Miles] ppo_utils 路径: {miles_ppo.__file__}")
    log_prob, _ = miles_ppo._calculate_log_probs_and_entropy_true_on_policy(
        logits, tokens, with_entropy=False
    )
    return log_prob


def hf_get_logprobs(
    model_path: str,
    token_ids: List[int],
    device: str,
    attn_implementation: str = "sdpa",
    use_batch_invariant: bool = False,
) -> torch.Tensor:
    """
    用 HuggingFace 模型做 teacher-forcing forward，模拟 Miles FSDP 的 logprob 计算。

    Returns:
        logprobs: shape (1, seq_len-1)，即对每个位置 i 预测 token[i+1] 的 logprob
    """
    if use_batch_invariant:
        try:
            from batch_invariant_trace import (
                enable_batch_invariant_mode_with_tracing,
                reset_call_counts,
            )
            reset_call_counts()
            enable_batch_invariant_mode_with_tracing()
        except ImportError as e:
            raise ImportError(
                " --use-batch-invariant 需要安装 sglang 且 batch_invariant_trace 在 PYTHONPATH 内: " + str(e)
            ) from e

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
    except ImportError:
        raise ImportError("需要安装 transformers: pip install transformers")

    if attn_implementation == "triton":
        from hf_triton_attention import triton_attention_forward
        AttentionInterface.register("triton", triton_attention_forward)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()

    ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)

    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # logits[:, :-1] 预测 tokens[:, 1:]，与 Miles get_responses + _calculate_log_probs 一致
    response_logits = logits[:, :-1, :].float()  # (1, seq_len-1, V)
    response_tokens = ids[:, 1:]  # (1, seq_len-1)

    if torch.isnan(response_logits).any() or torch.isinf(response_logits).any():
        print("WARN: logits 含 nan/inf，batch_invariant 可能有问题")

    logprobs = _compute_log_probs_miles_style(response_logits, response_tokens)

    return logprobs  # (1, seq_len-1)


# =============================================================================
# 4. 对齐比较
# =============================================================================


def ensure_token_texts(tokenizer, gen_token_ids: List[int], gen_token_texts: List[str]) -> List[str]:
    """若 SGLang 未返回 token 文本，用 tokenizer 解码补充。"""
    result = list(gen_token_texts) if len(gen_token_texts) >= len(gen_token_ids) else [""] * len(gen_token_ids)
    for i, tid in enumerate(gen_token_ids):
        if i < len(result) and result[i]:
            continue
        try:
            result[i] = tokenizer.decode([tid], skip_special_tokens=False)
            if not result[i] or result[i].isspace():
                result[i] = f"<{tid}>"
        except Exception:
            result[i] = f"<{tid}>"
    return result[: len(gen_token_ids)]


def align_and_compare(
    prompt_len: int,
    gen_logprobs_sglang: List[float],
    logprobs_hf: torch.Tensor,
    gen_token_ids: List[int],
    gen_token_texts: List[str],
    tolerance: float,
) -> Tuple[bool, List[dict]]:
    """
    比较 SGLang 与 HF 的 logprob。

    HF 的 logprobs[0, prompt_len-1 : prompt_len-1+N] 对应预测 gen_tokens 的 logprob。
    """
    n_gen = len(gen_logprobs_sglang)
    hf_slice = logprobs_hf[0, prompt_len - 1 : prompt_len - 1 + n_gen]

    if hf_slice.shape[0] != n_gen:
        return False, [{"error": f"长度不一致: HF={hf_slice.shape[0]}, SGLang={n_gen}"}]

    details = []
    all_match = True
    for i in range(n_gen):
        lp_sglang = gen_logprobs_sglang[i]
        lp_hf = hf_slice[i].item()
        diff = abs(lp_sglang - lp_hf)
        match = diff <= tolerance
        if not match:
            all_match = False
        details.append(
            {
                "pos": i,
                "token_id": gen_token_ids[i],
                "token_text": gen_token_texts[i] if i < len(gen_token_texts) else "",
                "logprob_sglang": lp_sglang,
                "logprob_hf": lp_hf,
                "diff": diff,
                "match": match,
            }
        )

    return all_match, details


# =============================================================================
# 5. 保存/加载 Rollout 到 txt（模拟 Miles：SGLang 存 → FSDP 读）
# =============================================================================


def save_rollout_to_txt(
    path: str,
    prompt: str,
    tokens: List[int],
    rollout_logprobs: List[float],
    prompt_len: int,
    gen_token_texts: List[str],
    model_path: str,
) -> None:
    """保存 SGLang rollout 信息到 txt（JSON 格式，便于解析）。"""
    data = {
        "prompt": prompt,
        "tokens": tokens,
        "rollout_logprobs": rollout_logprobs,
        "prompt_len": prompt_len,
        "gen_token_texts": gen_token_texts,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[保存] Rollout 已写入: {path}")


def load_rollout_from_txt(path: str) -> dict:
    """从 txt 加载 rollout 数据。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ["tokens", "rollout_logprobs", "prompt_len"]:
        if key not in data:
            raise ValueError(f"rollout 文件缺少字段: {key}")
    if "gen_token_texts" not in data:
        data["gen_token_texts"] = []
    return data


# =============================================================================
# 6. 保存人工对比详情
# =============================================================================


def _save_detail_file(
    path: str,
    args,
    prompt_len: int,
    gen_token_ids: List[int],
    details: List[dict],
) -> None:
    """保存 SGLang 与 HF 逐 token 对比，便于肉眼查看和学习。"""
    attn_impl = getattr(args, "attn_implementation", "sdpa")
    lines = [
        "=" * 80,
        "FSDP vs SGLang Logprob 人工对比详情",
        "=" * 80,
        f"Prompt: {args.prompt}",
        f"Prompt token 数: {prompt_len}",
        f"生成 token 数: {len(gen_token_ids)}",
        f"HF attn_implementation: {attn_impl}",
        f"容差: {args.tolerance}",
        "",
        "说明：",
        "  - pos: 生成 token 的位置（0=第一个生成 token）",
        "  - token_id: token 在词表中的 ID",
        "  - token_text: token 解码后的文本",
        "  - logprob_sglang: SGLang 推理返回的 logprob",
        "  - logprob_hf: HuggingFace forward 模拟 FSDP 的 logprob",
        "  - diff: 两者差的绝对值；1e-2~1e-4 的差异通常由 attention 实现/精度不同导致，属正常",
        "",
        "-" * 80,
        f"{'pos':>4} | {'token_id':>10} | {'token_text':<12} | {'SGLang':>12} | {'HF':>12} | {'diff':>10} | ok",
        "-" * 80,
    ]
    for d in details:
        if "error" in d:
            lines.append(f"错误: {d['error']}")
            break
        txt = (d.get("token_text", "") or "")[:12].replace("\n", " ")
        status = "✓" if d["match"] else "✗"
        lines.append(
            f"{d['pos']:4d} | {d['token_id']:10d} | {txt:<12} | "
            f"{d['logprob_sglang']:12.8f} | {d['logprob_hf']:12.8f} | "
            f"{d['diff']:10.2e} | {status}"
        )
    lines.append("-" * 80)

    # 计算 mean(|diff|) 并追加到文件末尾
    diffs = [d["diff"] for d in details if "diff" in d]
    if diffs:
        mean_abs_diff = sum(diffs) / len(diffs)
        metric_line = f"[metric] mean(|diff|) = {mean_abs_diff:.6e}  (n={len(diffs)})"
        lines.append("")
        lines.append(metric_line)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# 7. 主流程
# =============================================================================


def main():
    args = parse_args()
    load_rollout = args.load_rollout is not None

    print("=" * 70)
    print("FSDP vs SGLang Logprob 对齐测试（SGLang 存 → FSDP 读 → teacher-forcing）")
    print("=" * 70)
    if load_rollout:
        print(f"  模式: 从文件加载 rollout (--load-rollout)")
        print(f"  文件: {args.load_rollout}")
    else:
        print(f"  SGLang: http://{args.host}:{args.port}")
    print(f"  模型:   {args.model_path}")
    print(f"  HF attn_implementation: {args.attn_implementation}")
    use_bi = getattr(args, "use_batch_invariant", False)
    if use_bi:
        print(f"  batch_invariant_ops:   是")
    else:
        print(f"  batch_invariant_ops:   否")
    print(f"  容差:   {args.tolerance}")
    if not load_rollout:
        print(f"  Prompt: {args.prompt[:50]}...")
        print(f"  生成数: {args.max_new_tokens} tokens")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("错误: 需要安装 transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 1. 获取 rollout 数据：从文件加载 或 调用 SGLang 生成
    if load_rollout:
        print("\n[1] 从文件加载 rollout 数据...")
        rollout_data = load_rollout_from_txt(args.load_rollout)
        full_token_ids = rollout_data["tokens"]
        gen_logprobs_sglang = rollout_data["rollout_logprobs"]
        prompt_len = rollout_data["prompt_len"]
        gen_token_texts = rollout_data.get("gen_token_texts", [""] * len(gen_logprobs_sglang))
        gen_token_ids = full_token_ids[prompt_len:]
        args.prompt = rollout_data.get("prompt", args.prompt)
        if "model_path" in rollout_data:
            args.model_path = rollout_data["model_path"]
        print(f"  完整序列长度: {len(full_token_ids)}")
        print(f"  Prompt token 数: {prompt_len}")
        print(f"  生成 token 数: {len(gen_token_ids)}")
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)
        print(f"\n[1] Prompt token 数: {prompt_len}")

        print("\n[2] 调用 SGLang 贪婪生成...")
        try:
            full_token_ids, gen_logprobs_sglang, gen_token_ids, gen_token_texts = (
                sglang_generate_with_logprobs(
                    args.host,
                    args.port,
                    prompt_ids,
                    args.max_new_tokens,
                )
            )
        except Exception as e:
            print(f"错误: SGLang 调用失败: {e}")
            sys.exit(1)

        print(f"  生成 {len(gen_token_ids)} 个 token")
        print(f"  完整序列长度: {len(full_token_ids)}")

        if args.save_rollout:
            save_rollout_to_txt(
                path=args.save_rollout,
                prompt=args.prompt,
                tokens=full_token_ids,
                rollout_logprobs=gen_logprobs_sglang,
                prompt_len=prompt_len,
                gen_token_texts=gen_token_texts,
                model_path=args.model_path,
            )

    # 若 SGLang 未返回 token 文本，用 tokenizer 解码补充（解决 token_text 为空）
    gen_token_texts = ensure_token_texts(tokenizer, gen_token_ids, gen_token_texts)

    # 2/3. FSDP 侧 teacher-forcing forward
    step = 2 if load_rollout else 3
    batch_invariant_str = " + batch_invariant_ops" if getattr(args, "use_batch_invariant", False) else ""
    print(f"\n[{step}] FSDP 侧 teacher-forcing forward（HF attn={args.attn_implementation}{batch_invariant_str}）...")

    try:
        logprobs_hf = hf_get_logprobs(
            args.model_path,
            full_token_ids,
            args.device,
            attn_implementation=args.attn_implementation,
            use_batch_invariant=getattr(args, "use_batch_invariant", False),
        )
    except Exception as e:
        print(f"错误: HF forward 失败: {e}")
        sys.exit(1)

    if getattr(args, "use_batch_invariant", False):
        from batch_invariant_trace import print_call_counts

        print_call_counts()

    # 3/4. 比对 logprob
    step_compare = 3 if load_rollout else 4
    print(f"\n[{step_compare}] 对齐比较...")
    all_match, details = align_and_compare(
        prompt_len,
        gen_logprobs_sglang,
        logprobs_hf,
        gen_token_ids,
        gen_token_texts,
        args.tolerance,
    )

    # 5. 输出
    print("\n" + "-" * 70)
    print("逐 token 对比（前 10 个）:")
    print("-" * 70)
    for d in details[:10]:
        status = "✓" if d["match"] else "✗"
        print(
            f"  {status} pos={d['pos']:2d}  SGLang={d['logprob_sglang']:.8f}  "
            f"HF={d['logprob_hf']:.8f}  diff={d['diff']:.2e}"
        )
    if len(details) > 10:
        print(f"  ... 共 {len(details)} 个 token")

    print("\n" + "=" * 70)
    if all_match:
        print("✓ PASS: SGLang 与 FSDP-style (HF) logprob 完全一致")
    else:
        mismatches = sum(1 for d in details if not d["match"])
        print(f"✗ FAIL: {mismatches}/{len(details)} 个 token logprob 不一致")
    print("=" * 70)

    # 6. 保存 JSON
    if args.output_json:
        config = {
            "model_path": args.model_path,
            "attn_implementation": args.attn_implementation,
            "tolerance": args.tolerance,
        }
        if load_rollout:
            config["load_rollout"] = args.load_rollout
        else:
            config["host"] = args.host
            config["port"] = args.port
            config["max_new_tokens"] = args.max_new_tokens
        result = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "prompt": args.prompt,
            "prompt_len": prompt_len,
            "gen_count": len(gen_token_ids),
            "pass": all_match,
            "details": details,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output_json}")

    # 7. 保存人工对比详情（便于肉眼对比和学习）
    detail_path = args.save_detail
    if detail_path is None and args.output_json:
        base = args.output_json.rsplit(".", 1)[0]
        detail_path = f"{base}_detail.txt"
    elif detail_path is None:
        # 默认保存到 results/，便于学习对比
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_path = f"results/align_detail_{args.attn_implementation}_{ts}.txt"

    if detail_path:
        _save_detail_file(detail_path, args, prompt_len, gen_token_ids, details)
        print(f"人工对比详情已保存: {detail_path}")

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()