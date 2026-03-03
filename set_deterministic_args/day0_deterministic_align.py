#!/usr/bin/env python3
"""
Day-0 deterministic alignment scaffold for true on-policy on AMD/NVIDIA.

Goals:
1) Lock down runtime/sampling/dtype/environment into an auditable manifest.
2) Compare rollout-side and training-side log_probs with numeric and bit-wise checks.
3) Provide a minimal, reproducible harness before wiring into full miles training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests
import torch


DEFAULT_ENV_KEYS = [
    "NCCL_ALGO",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "PYTORCH_DETERMINISTIC",
]


@dataclass
class CompareSummary:
    total: int
    numeric_match_count: int
    bitwise_match_count: int
    first_numeric_mismatch: int | None
    first_bitwise_mismatch: int | None
    numeric_all_match: bool
    bitwise_all_match: bool


_dumper = None


def _maybe_dump(name: str, value: torch.Tensor) -> None:
    global _dumper
    if _dumper is False:
        return
    if _dumper is None:
        try:
            _dumper = __import__(
                "sglang.srt.debug_utils.dumper",
                fromlist=["dumper"],
            ).dumper
        except Exception:
            _dumper = False
            return
    _dumper.dump(name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day-0 deterministic alignment: SGLang rollout vs HF(FSDP-style) log_probs."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3", "triton"],
    )
    parser.add_argument("--prompt", type=str, default="请简单介绍下 SGLang，它是什么、能做什么。")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # Explicitly pin all sampling knobs to avoid hidden defaults.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)

    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--strict-bitwise", action="store_true")
    parser.add_argument("--use-batch-invariant", action="store_true")

    parser.add_argument("--save-rollout", type=str, default=None)
    parser.add_argument("--load-rollout", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--save-detail", type=str, default=None)
    parser.add_argument("--manifest-json", type=str, default=None)
    return parser.parse_args()


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "min_p": float(args.min_p),
        "max_new_tokens": int(args.max_new_tokens),
        "repetition_penalty": float(args.repetition_penalty),
        "frequency_penalty": float(args.frequency_penalty),
        "presence_penalty": float(args.presence_penalty),
        "sampling_seed": int(args.seed),
    }


def env_snapshot(keys: Iterable[str] = DEFAULT_ENV_KEYS) -> dict[str, str]:
    return {k: os.environ.get(k, "") for k in keys}


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_manifest(path: str | Path, args: argparse.Namespace, prompt_ids: list[int]) -> None:
    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "args": vars(args),
        "prompt_ids": prompt_ids,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": device_name,
        },
        "env": env_snapshot(),
    }
    save_json(path, payload)


def _load_rollout(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["tokens", "rollout_logprobs", "prompt_len", "gen_token_ids", "gen_token_texts"]
    for key in required:
        if key not in data:
            raise ValueError(f"Rollout file missing field: {key}")
    return data


def _save_rollout(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def _to_float32_int_view(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    return x.contiguous().view(torch.int32)


def first_mismatch_index(lhs: torch.Tensor, rhs: torch.Tensor) -> int | None:
    if lhs.shape != rhs.shape:
        return 0
    l = lhs.reshape(-1)
    r = rhs.reshape(-1)
    neq = torch.nonzero(l != r, as_tuple=False)
    if neq.numel() == 0:
        return None
    return int(neq[0].item())


def compare_logprobs(
    sglang_logprobs: list[float],
    hf_logprobs: torch.Tensor,
    tolerance: float,
) -> tuple[list[dict[str, Any]], CompareSummary]:
    hf = hf_logprobs.detach().to(torch.float32).cpu().reshape(-1)
    sg = torch.tensor(sglang_logprobs, dtype=torch.float32).reshape(-1)
    if hf.shape[0] != sg.shape[0]:
        raise ValueError(f"Length mismatch: hf={hf.shape[0]}, sglang={sg.shape[0]}")

    hf_i = _to_float32_int_view(hf)
    sg_i = _to_float32_int_view(sg)
    details: list[dict[str, Any]] = []
    numeric_match_count = 0
    bitwise_match_count = 0
    first_numeric_mismatch = None
    first_bitwise_mismatch = None

    for i in range(hf.shape[0]):
        a = float(sg[i].item())
        b = float(hf[i].item())
        diff = abs(a - b)
        numeric_match = diff <= tolerance
        bitwise_match = int(sg_i[i].item()) == int(hf_i[i].item())
        if numeric_match:
            numeric_match_count += 1
        elif first_numeric_mismatch is None:
            first_numeric_mismatch = i
        if bitwise_match:
            bitwise_match_count += 1
        elif first_bitwise_mismatch is None:
            first_bitwise_mismatch = i
        details.append(
            {
                "pos": i,
                "logprob_sglang": a,
                "logprob_hf": b,
                "abs_diff": diff,
                "numeric_match": numeric_match,
                "bitwise_match": bitwise_match,
            }
        )

    summary = CompareSummary(
        total=hf.shape[0],
        numeric_match_count=numeric_match_count,
        bitwise_match_count=bitwise_match_count,
        first_numeric_mismatch=first_numeric_mismatch,
        first_bitwise_mismatch=first_bitwise_mismatch,
        numeric_all_match=numeric_match_count == hf.shape[0],
        bitwise_all_match=bitwise_match_count == hf.shape[0],
    )
    return details, summary


def _save_detail(path: str | Path, details: list[dict[str, Any]], summary: CompareSummary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 88)
    lines.append("Day-0 deterministic align detail")
    lines.append("=" * 88)
    lines.append(
        f"total={summary.total}, numeric={summary.numeric_match_count}/{summary.total}, "
        f"bitwise={summary.bitwise_match_count}/{summary.total}"
    )
    lines.append(
        f"first_numeric_mismatch={summary.first_numeric_mismatch}, "
        f"first_bitwise_mismatch={summary.first_bitwise_mismatch}"
    )
    lines.append("-" * 88)
    lines.append(f"{'pos':>4} | {'sglang':>12} | {'hf':>12} | {'abs_diff':>12} | {'num':>3} | {'bit':>3}")
    lines.append("-" * 88)
    for d in details:
        lines.append(
            f"{d['pos']:4d} | {d['logprob_sglang']:12.8f} | {d['logprob_hf']:12.8f} | "
            f"{d['abs_diff']:12.4e} | {'Y' if d['numeric_match'] else 'N':>3} | "
            f"{'Y' if d['bitwise_match'] else 'N':>3}"
        )
    lines.append("-" * 88)
    path.write_text("\n".join(lines), encoding="utf-8")


def sglang_generate_with_logprobs(
    host: str,
    port: int,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> tuple[list[int], list[float], list[int], list[str]]:
    base_url = f"http://{host}:{port}"
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "stream": False,
    }
    response = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"SGLang request failed: {response.status_code} {response.text}")
    ret = response.json()
    if isinstance(ret, list):
        ret = ret[0]

    output_ids = ret["output_ids"]
    meta = ret.get("meta_info", {})
    token_lps = meta.get("output_token_logprobs", [])
    if not token_lps:
        raise RuntimeError("No output_token_logprobs returned by SGLang.")

    n = len(output_ids)
    gen_logprobs = [float(x[0]) for x in token_lps[:n]]
    gen_texts = [x[2] if len(x) > 2 else "" for x in token_lps[:n]]
    full_ids = prompt_ids + output_ids
    return full_ids, gen_logprobs, output_ids, gen_texts


def hf_get_logprobs(
    model_path: str,
    token_ids: list[int],
    device: str,
    attn_implementation: str,
    dtype: str,
    use_batch_invariant: bool,
) -> torch.Tensor:
    if use_batch_invariant:
        try:
            from batch_invariant_trace import enable_batch_invariant_mode_with_tracing
        except ImportError as e:
            raise ImportError("Failed to import batch_invariant_trace; install sglang and PYTHONPATH first.") from e
        enable_batch_invariant_mode_with_tracing()

    from transformers import AttentionInterface, AutoModelForCausalLM

    if attn_implementation == "triton":
        attention_dir = Path(__file__).resolve().parent.parent / "attention_test"
        if str(attention_dir) not in sys.path:
            sys.path.insert(0, str(attention_dir))
        from hf_triton_attention import triton_attention_forward

        AttentionInterface.register("triton", triton_attention_forward)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype),
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    _maybe_dump("input_ids_for_compare", ids)
    input_embeds = None
    try:
        # Align with SGLang rl_on_policy path: feed FP32 embeddings into pre-attn path.
        input_embeds = model.get_input_embeddings()(ids).float()
        _maybe_dump("embedding_output", input_embeds)
    except Exception:
        pass

    def _hf_slice(x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return x
        if x.ndim >= 3:
            return x[:, :-1, ...]
        return x

    attn_out_last_layer = None
    hook_handles = []
    layer0_positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.long).unsqueeze(0)
    _maybe_dump("layer0_positions", layer0_positions)

    # Align HF compute dtype with SGLang:
    # 1) Keep pre-attn norm/residual path in FP32.
    # 2) Cast hidden_states to BF16 right before self_attn.
    def _norm_pre_fp32(_module, args, kwargs):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            new_args = (args[0].float(),) + tuple(args[1:])
            return new_args, kwargs
        return None

    def _norm_post_fp32(_module, args, kwargs, output):
        if isinstance(output, torch.Tensor):
            return output.float()
        return output

    def _self_attn_pre_bf16(_module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        if hs is None or not isinstance(hs, torch.Tensor):
            return None
        # Capture "after prepare_attn, before self_attn bf16 cast" for layer-0.
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                first_layer = model.model.layers[0] if len(model.model.layers) > 0 else None
                if first_layer is not None and getattr(first_layer, "self_attn", None) is _module:
                    _maybe_dump("layer0_attn_input_after_prepare", _hf_slice(hs))
        except Exception:
            pass
        hs = hs.to(torch.bfloat16)
        if "hidden_states" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hs
            return args, new_kwargs
        new_args = list(args)
        new_args[0] = hs
        return tuple(new_args), kwargs

    def _mlp_pre_bf16(_module, args, kwargs):
        x = kwargs.get("hidden_states", kwargs.get("x", args[0] if len(args) > 0 else None))
        if x is None or not isinstance(x, torch.Tensor):
            return None
        x = x.to(torch.bfloat16)
        if "hidden_states" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = x
            return args, new_kwargs
        if "x" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["x"] = x
            return args, new_kwargs
        new_args = list(args)
        new_args[0] = x
        return tuple(new_args), kwargs

    def _o_proj_pre_cast(_module, args, kwargs):
        if len(args) == 0 or not isinstance(args[0], torch.Tensor):
            return None
        x = args[0]
        target_dtype = _module.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return (x,) + tuple(args[1:]), kwargs

    def _lm_head_pre_cast(_module, args, kwargs):
        if len(args) == 0 or not isinstance(args[0], torch.Tensor):
            return None
        x = args[0]
        target_dtype = _module.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return (x,) + tuple(args[1:]), kwargs

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                norm_mod = getattr(layer, norm_name, None)
                if norm_mod is not None:
                    hook_handles.append(
                        norm_mod.register_forward_pre_hook(_norm_pre_fp32, with_kwargs=True)
                    )
                    hook_handles.append(
                        norm_mod.register_forward_hook(_norm_post_fp32, with_kwargs=True)
                    )
            if hasattr(layer, "self_attn"):
                hook_handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        _self_attn_pre_bf16, with_kwargs=True
                    )
                )
                if hasattr(layer.self_attn, "o_proj"):
                    hook_handles.append(
                        layer.self_attn.o_proj.register_forward_pre_hook(
                            _o_proj_pre_cast, with_kwargs=True
                        )
                    )
            if hasattr(layer, "mlp"):
                hook_handles.append(
                    layer.mlp.register_forward_pre_hook(
                        _mlp_pre_bf16, with_kwargs=True
                    )
                )
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            hook_handles.append(
                model.lm_head.register_forward_pre_hook(
                    _lm_head_pre_cast, with_kwargs=True
                )
            )

    # Layer-0 pre-attention probes.
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        first_layer = model.model.layers[0]

        def _capture_layer0_raw(_module, args, kwargs):
            try:
                hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                if hs is not None:
                    _maybe_dump("layer0_attn_input_raw", _hf_slice(hs))
            except Exception:
                pass

        hook_handles.append(
            first_layer.register_forward_pre_hook(_capture_layer0_raw, with_kwargs=True)
        )

        # layer0_attn_input_after_prepare is captured inside _self_attn_pre_bf16
        # before bf16 cast, to match SGLang's semantic timing.

    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        last_layer = model.model.layers[-1]
        if hasattr(last_layer, "self_attn"):
            def _capture_last_attn_output(_module, args, kwargs, output):
                nonlocal attn_out_last_layer
                attn_out_last_layer = output[0] if isinstance(output, tuple) else output
                try:
                    hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                    if hidden_states is None:
                        return
                    _maybe_dump("attn_input_last_layer", _hf_slice(hidden_states))

                    if hasattr(_module, "q_proj") and hasattr(_module, "k_proj") and hasattr(_module, "v_proj"):
                        q = _module.q_proj(hidden_states)
                        k = _module.k_proj(hidden_states)
                        v = _module.v_proj(hidden_states)
                        _maybe_dump("q_pre_norm", _hf_slice(q))
                        _maybe_dump("k_pre_norm", _hf_slice(k))
                        _maybe_dump("v_pre_norm", _hf_slice(v))

                        qn, kn = q, k
                        if hasattr(_module, "q_norm") and hasattr(_module, "k_norm"):
                            num_heads = getattr(_module, "num_heads", None)
                            num_kv_heads = getattr(_module, "num_key_value_heads", num_heads)
                            if num_heads is not None and num_kv_heads is not None:
                                q_head_dim = q.shape[-1] // num_heads
                                k_head_dim = k.shape[-1] // num_kv_heads
                                qn = _module.q_norm(q.view(q.shape[0], q.shape[1], num_heads, q_head_dim)).reshape_as(q)
                                kn = _module.k_norm(k.view(k.shape[0], k.shape[1], num_kv_heads, k_head_dim)).reshape_as(k)
                                _maybe_dump("q_post_norm", _hf_slice(qn))
                                _maybe_dump("k_post_norm", _hf_slice(kn))

                    if attn_out_last_layer is not None:
                        _maybe_dump("attn_context_before_o_proj", _hf_slice(attn_out_last_layer))
                except Exception:
                    pass

            hook_handles.append(
                last_layer.self_attn.register_forward_hook(
                    _capture_last_attn_output, with_kwargs=True
                )
            )

    with torch.no_grad():
        if input_embeds is not None:
            outputs = model(
                input_ids=None,
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = model(ids, output_hidden_states=True, return_dict=True)
        logits = outputs.logits
        final_hidden_before_lm_head = outputs.hidden_states[-1]
        if attn_out_last_layer is not None:
            _maybe_dump("attn_out_last_layer", attn_out_last_layer[:, :-1, :])
        _maybe_dump(
            "final_hidden_before_lm_head",
            final_hidden_before_lm_head[:, :-1, :],
        )
        _maybe_dump("lm_head_weight", model.lm_head.weight)
    for h in hook_handles:
        try:
            h.remove()
        except Exception:
            pass

    response_logits = logits[:, :-1, :].float()
    response_tokens = ids[:, 1:]

    miles_dir = Path(__file__).resolve().parents[2] / "miles"
    if str(miles_dir) not in sys.path:
        sys.path.insert(0, str(miles_dir))
    import miles.utils.ppo_utils as ppo_utils

    log_probs, _ = ppo_utils._calculate_log_probs_and_entropy_true_on_policy(
        response_logits, response_tokens, with_entropy=False
    )
    return log_probs


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    if args.manifest_json is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.manifest_json = f"results/day0_manifest_{ts}.json"
    save_manifest(args.manifest_json, args, prompt_ids)

    if args.load_rollout is not None:
        rollout = _load_rollout(args.load_rollout)
        full_token_ids = rollout["tokens"]
        prompt_len = int(rollout["prompt_len"])
        gen_logprobs_sglang = rollout["rollout_logprobs"]
        gen_token_ids = rollout["gen_token_ids"]
        gen_token_texts = rollout["gen_token_texts"]
        sampling_params = rollout.get("sampling_params", build_sampling_params(args))
    else:
        sampling_params = build_sampling_params(args)
        full_token_ids, gen_logprobs_sglang, gen_token_ids, gen_token_texts = sglang_generate_with_logprobs(
            args.host,
            args.port,
            prompt_ids,
            sampling_params,
        )
        if args.save_rollout:
            _save_rollout(
                args.save_rollout,
                {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": args.model_path,
                    "prompt": args.prompt,
                    "prompt_len": prompt_len,
                    "tokens": full_token_ids,
                    "gen_token_ids": gen_token_ids,
                    "gen_token_texts": gen_token_texts,
                    "rollout_logprobs": gen_logprobs_sglang,
                    "sampling_params": sampling_params,
                },
            )

    hf_logprobs_full = hf_get_logprobs(
        model_path=args.model_path,
        token_ids=full_token_ids,
        device=args.device,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        use_batch_invariant=args.use_batch_invariant,
    )
    n_gen = len(gen_token_ids)
    hf_slice = hf_logprobs_full[0, prompt_len - 1 : prompt_len - 1 + n_gen]

    details, summary = compare_logprobs(
        sglang_logprobs=gen_logprobs_sglang,
        hf_logprobs=hf_slice,
        tolerance=args.tolerance,
    )
    token_equal = list(full_token_ids[prompt_len:]) == list(gen_token_ids)

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": args.model_path,
            "attn_implementation": args.attn_implementation,
            "dtype": args.dtype,
            "seed": args.seed,
            "sampling_params": sampling_params,
            "tolerance": args.tolerance,
            "strict_bitwise": args.strict_bitwise,
        },
        "checks": {
            "token_ids_equal": token_equal,
            "numeric_all_match": summary.numeric_all_match,
            "bitwise_all_match": summary.bitwise_all_match,
            "first_numeric_mismatch": summary.first_numeric_mismatch,
            "first_bitwise_mismatch": summary.first_bitwise_mismatch,
        },
        "summary": {
            "total": summary.total,
            "numeric_match_count": summary.numeric_match_count,
            "bitwise_match_count": summary.bitwise_match_count,
        },
        "details": details,
    }

    if args.output_json is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = f"results/day0_compare_{ts}.json"
    save_json(args.output_json, result)

    if args.save_detail is None:
        args.save_detail = str(Path(args.output_json).with_suffix("")) + "_detail.txt"
    _save_detail(args.save_detail, details, summary)

    print("=" * 80)
    print("Day-0 deterministic compare")
    print("=" * 80)
    print(f"token_ids_equal:      {token_equal}")
    print(f"numeric_all_match:    {summary.numeric_all_match}")
    print(f"bitwise_all_match:    {summary.bitwise_all_match}")
    print(f"first_numeric_mismatch: {summary.first_numeric_mismatch}")
    print(f"first_bitwise_mismatch: {summary.first_bitwise_mismatch}")
    print(f"manifest: {args.manifest_json}")
    print(f"result:   {args.output_json}")
    print(f"detail:   {args.save_detail}")
    print("=" * 80)

    passed = summary.bitwise_all_match if args.strict_bitwise else summary.numeric_all_match
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
