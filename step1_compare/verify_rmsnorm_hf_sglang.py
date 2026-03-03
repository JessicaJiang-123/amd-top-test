#!/usr/bin/env python3
"""Compare HF Qwen3RMSNorm vs SGLang RMSNorm on the same dumped input."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

SG_INDEX_PRESET_STEP3_ATTN = {
    "input_ids_for_compare": 1,
    "embedding_output": 2,
    "layer0_attn_input_raw": 3,
    "layer0_positions": 4,
    # RMSNorm stage probes (prefill first pass)
    "rmsnorm_stage_x_fp32": 5,
    "rmsnorm_stage_variance": 6,
    "rmsnorm_stage_x_norm_fp32": 7,
    "rmsnorm_stage_x_out": 8,
    # New probe inserted right after input_layernorm and before communication.
    "layer0_attn_after_input_layernorm_only": 9,
    "layer0_attn_input_after_prepare": 10,
    "attn_input_last_layer": 11,
    "q_pre_norm": 12,
    "k_pre_norm": 13,
    "v_pre_norm": 14,
    "q_post_norm": 15,
    "k_post_norm": 16,
    "q_post_rope": 17,
    "k_post_rope": 18,
    "attn_context_before_o_proj": 19,
    "attn_out_last_layer": 20,
    "final_hidden_before_lm_head": 21,
    "lm_head_weight": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use SGLang dumped layer0 input/output around prepare_attn as ground truth, "
            "then compare local HF RMSNorm and local SGLang RMSNorm outputs."
        )
    )
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--input-name", type=str, default="layer0_attn_input_raw")
    parser.add_argument(
        "--output-name",
        type=str,
        default="layer0_attn_after_input_layernorm_only",
    )
    parser.add_argument("--input-index", type=int, default=None)
    parser.add_argument("--output-index", type=int, default=None)
    parser.add_argument(
        "--index-source",
        type=str,
        default="step3_sg_index",
        choices=["step3_sg_index", "explicit_or_auto"],
        help=(
            "How to resolve dump index when --input-index/--output-index is omitted: "
            "'step3_sg_index' uses compare_hf_sglang_step3_attn.py SG_INDEX preset; "
            "'explicit_or_auto' uses auto-index-policy."
        ),
    )
    parser.add_argument(
        "--auto-index-policy",
        type=str,
        default="min",
        choices=["error", "min", "max"],
        help=(
            "When index is not provided and multiple dump files exist: "
            "'error' -> raise; 'min' -> pick smallest dump_index; 'max' -> pick largest dump_index."
        ),
    )
    parser.add_argument(
        "--weight-key",
        type=str,
        default="model.layers.0.input_layernorm.weight",
        help="HF state_dict key for layer0 input_layernorm weight.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--sg-call",
        type=str,
        default="forward_native",
        choices=["forward_native", "forward"],
        help="How to call SGLang RMSNorm locally.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--compare-stages",
        action="store_true",
        help="Compare dumped RMSNorm intermediate stages against local recomputation.",
    )
    parser.add_argument(
        "--stage-prefix",
        type=str,
        default="rmsnorm_stage",
        help="Prefix for RMSNorm intermediate dump names.",
    )
    parser.add_argument(
        "--profile-local",
        action="store_true",
        help="Profile local RMSNorm execution and save operator/kernel reports.",
    )
    parser.add_argument(
        "--profile-out-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for local profiler reports.",
    )
    return parser.parse_args()


def _extract_dump_index(path: Path) -> int:
    match = re.search(r"___dump_index=(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse dump_index from file name: {path.name}")
    return int(match.group(1))


def resolve_unique_dump_file(
    dump_dir: Path,
    name: str,
    index: int | None,
    auto_index_policy: str = "error",
) -> Path:
    if index is not None:
        path = dump_dir / f"forward_pass_id=0___rank=0___name={name}___dump_index={index}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing dump file: {path}")
        return path

    matches = [
        path
        for path in dump_dir.glob(f"forward_pass_id=0___rank=0___name={name}___dump_index=*.pt")
        if path.is_file()
    ]
    matches = sorted(matches, key=_extract_dump_index)
    if not matches:
        raise FileNotFoundError(f"No dump file matched name={name} under {dump_dir}")
    if len(matches) > 1:
        if auto_index_policy == "error":
            raise RuntimeError(
                f"Multiple dump files matched name={name} under {dump_dir}: "
                + ", ".join(path.name for path in matches)
            )
        if auto_index_policy == "min":
            return matches[0]
        if auto_index_policy == "max":
            return matches[-1]
        raise ValueError(f"Unsupported auto_index_policy: {auto_index_policy}")
    return matches[0]


def load_dump_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, weights_only=False, map_location="cpu")
    value = obj["value"] if isinstance(obj, dict) and "value" in obj else obj
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Dump {path} did not contain a tensor, got {type(value).__name__}")
    return value


def print_tensor_meta(label: str, value: torch.Tensor) -> None:
    print(
        f"{label}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
        f"stride={tuple(value.stride())} contiguous={value.is_contiguous()}"
    )


def validate_hidden_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    # Keep dump tensor rank exactly as dumped by server.
    # For RMSNorm we only require last dim == hidden and rank in {2, 3}.
    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"{name} expected rank-2 or rank-3 hidden tensor, got shape={tuple(tensor.shape)}"
        )
    return tensor


def compare_pair(label: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float | None, float | None, bool]:
    print(f"\n[{label}]")
    print("  lhs shape/dtype:", tuple(lhs.shape), lhs.dtype)
    print("  rhs shape/dtype:", tuple(rhs.shape), rhs.dtype)
    if lhs.shape != rhs.shape:
        print("  -> shape mismatch")
        return None, None, False
    diff = (lhs.float() - rhs.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    bitwise_equal = lhs.dtype == rhs.dtype and torch.equal(lhs.contiguous(), rhs.contiguous())
    print("  max_abs:", max_abs)
    print("  mean_abs:", mean_abs)
    print("  bitwise_equal:", bitwise_equal)
    return max_abs, mean_abs, bitwise_equal


def profile_once(
    *,
    label: str,
    fn,
    x: torch.Tensor,
    out_dir: Path,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        _ = fn(x.clone())

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{label}.txt"
    trace_path = out_dir / f"{label}.json"
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total" if x.is_cuda else "self_cpu_time_total",
        row_limit=200,
    )
    table_path.write_text(table, encoding="utf-8")
    prof.export_chrome_trace(str(trace_path))
    print(f"[profile] {label} table={table_path}")
    print(f"[profile] {label} trace={trace_path}")


class HfQwen3RMSNorm(nn.Module):
    """Mirror of transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def resolve_model_dir(model_path: str) -> Path:
    p = Path(model_path)
    if p.exists():
        return p

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=model_path,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    return Path(local_dir)


def load_weight_from_model(model_path: str, weight_key: str) -> torch.Tensor:
    model_dir = resolve_model_dir(model_path)

    single_file = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"

    from safetensors import safe_open

    if single_file.exists():
        with safe_open(str(single_file), framework="pt", device="cpu") as f:
            if weight_key not in f.keys():
                raise KeyError(f"{weight_key} not found in {single_file}")
            return f.get_tensor(weight_key)

    if index_file.exists():
        with index_file.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if weight_key not in weight_map:
            raise KeyError(f"{weight_key} not found in {index_file}")
        shard = model_dir / weight_map[weight_key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            return f.get_tensor(weight_key)

    raise FileNotFoundError(
        f"Cannot locate safetensors files under {model_dir}. "
        "Expected model.safetensors or model.safetensors.index.json"
    )


def main() -> None:
    args = parse_args()

    os.environ["SGLANG_USE_AITER"] = "0"
    if "/app/sglang/python" not in sys.path:
        sys.path.insert(0, "/app/sglang/python")

    from sglang.srt.layers.layernorm import RMSNorm as SgRMSNorm
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    input_index = args.input_index
    output_index = args.output_index
    input_index_source = "explicit"
    output_index_source = "explicit"

    if args.index_source == "step3_sg_index":
        if input_index is None and args.input_name in SG_INDEX_PRESET_STEP3_ATTN:
            input_index = SG_INDEX_PRESET_STEP3_ATTN[args.input_name]
            input_index_source = "step3_sg_index_preset"
        elif input_index is None:
            input_index_source = "auto_policy_fallback"
        if output_index is None and args.output_name in SG_INDEX_PRESET_STEP3_ATTN:
            output_index = SG_INDEX_PRESET_STEP3_ATTN[args.output_name]
            output_index_source = "step3_sg_index_preset"
        elif output_index is None:
            output_index_source = "auto_policy_fallback"
    else:
        if input_index is None:
            input_index_source = "auto_policy"
        if output_index is None:
            output_index_source = "auto_policy"

    in_path = resolve_unique_dump_file(
        args.dump_dir,
        args.input_name,
        input_index,
        auto_index_policy=args.auto_index_policy,
    )
    out_path = resolve_unique_dump_file(
        args.dump_dir,
        args.output_name,
        output_index,
        auto_index_policy=args.auto_index_policy,
    )

    x_in = load_dump_tensor(in_path)
    y_dump = load_dump_tensor(out_path)

    if args.verbose:
        print(f"index_source={args.index_source}")
        print(f"resolved_input_index={_extract_dump_index(in_path)} ({input_index_source})")
        print(f"resolved_output_index={_extract_dump_index(out_path)} ({output_index_source})")
        print(f"resolved_input_path={in_path}")
        print(f"resolved_output_path={out_path}")
        print_tensor_meta("input_dump_raw", x_in)
        print_tensor_meta("output_dump_raw", y_dump)

    x_in = validate_hidden_tensor(args.input_name, x_in)
    y_dump = validate_hidden_tensor(args.output_name, y_dump)

    if x_in.shape != y_dump.shape:
        raise ValueError(
            f"Input/output shape mismatch: input={tuple(x_in.shape)} output={tuple(y_dump.shape)}"
        )

    hidden_size = x_in.shape[-1]
    weight = load_weight_from_model(args.model_path, args.weight_key).contiguous()
    if weight.ndim != 1 or weight.shape[0] != hidden_size:
        raise ValueError(
            f"Weight shape mismatch: key={args.weight_key} has {tuple(weight.shape)}, expected ({hidden_size},)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_in = x_in.contiguous().to(device=device, dtype=torch.float32)
    y_dump = y_dump.contiguous().to(device)

    # RMSNorm.__init__ will access global server args to decide on-policy behavior.
    set_global_server_args_for_scheduler(
        SimpleNamespace(rl_on_policy_target="fsdp" if device.type == "cuda" else None)
    )

    # HF RMSNorm branch.
    hf_norm = HfQwen3RMSNorm(hidden_size=hidden_size, eps=args.eps).to(device)
    hf_norm.weight.data.copy_(weight.to(device=device, dtype=hf_norm.weight.dtype))
    y_hf = hf_norm(x_in.clone())

    # SGLang RMSNorm branch (matching qwen3.py on-policy input_layernorm kwargs).
    sg_norm = SgRMSNorm(
        hidden_size=hidden_size,
        eps=args.eps,
        weight_dtype=torch.float32,
        cast_x_before_out_mul=True,
        override_orig_dtype=torch.float32,
        fp32_residual=True,
    ).to(device)
    sg_norm.weight.data.copy_(weight.to(device=device, dtype=sg_norm.weight.dtype))
    if args.sg_call == "forward_native":
        y_sg = sg_norm.forward_native(x_in.clone())
    else:
        y_sg = sg_norm(x_in.clone())

    if isinstance(y_sg, tuple):
        y_sg = y_sg[0]

    if args.verbose:
        print(f"hf_eps={hf_norm.variance_epsilon}")
        print(f"sg_eps={sg_norm.variance_epsilon}")
        print(
            "sg_norm_cfg:",
            f"cast_x_before_out_mul={sg_norm.cast_x_before_out_mul},",
            f"override_orig_dtype={sg_norm.override_orig_dtype},",
            f"variance_size_override={sg_norm.variance_size_override},",
            f"weight_dtype={sg_norm.weight.dtype}",
        )

    if args.profile_local:
        profile_once(
            label="local_profile_hf_real",
            fn=lambda z: hf_norm(z),
            x=x_in,
            out_dir=args.profile_out_dir,
        )
        profile_once(
            label="local_profile_sg_forward_native",
            fn=lambda z: sg_norm.forward_native(z),
            x=x_in,
            out_dir=args.profile_out_dir,
        )

    y_hf = y_hf.to(y_dump.dtype)
    y_sg = y_sg.to(y_dump.dtype)

    if args.verbose:
        print(f"execution_device={device}")
        print(f"model_path={args.model_path}")
        print(f"weight_key={args.weight_key}")
        print(f"loaded_weight_dtype={weight.dtype}")
        print(f"sg_call={args.sg_call}")
        print_tensor_meta("input_aligned_to_dump", x_in)
        print_tensor_meta("dump_output_aligned", y_dump)
        print_tensor_meta("hf_output_local", y_hf)
        print_tensor_meta("sg_output_local", y_sg)

    # Compare 1: HF local vs SGLang local
    hf_sg_max, hf_sg_mean, hf_sg_bit = compare_pair("hf_local_vs_sg_local", y_hf.cpu(), y_sg.cpu())

    # Compare 2: HF local vs SG server dump
    hf_dump_max, hf_dump_mean, hf_dump_bit = compare_pair("hf_local_vs_sg_dump", y_hf.cpu(), y_dump.cpu())

    # Compare 3: SGLang local vs SG server dump
    sg_dump_max, sg_dump_mean, sg_dump_bit = compare_pair("sg_local_vs_sg_dump", y_sg.cpu(), y_dump.cpu())

    # Compare 4: delta from input
    compare_pair("delta_hf_vs_dump", (y_hf - x_in.to(y_hf.dtype)).cpu(), (y_dump - x_in.to(y_dump.dtype)).cpu())
    compare_pair("delta_sg_vs_dump", (y_sg - x_in.to(y_sg.dtype)).cpu(), (y_dump - x_in.to(y_dump.dtype)).cpu())

    if args.compare_stages:
        stage_defs = [
            ("x_fp32", x_in.to(torch.float32)),
            ("variance", x_in.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)),
        ]
        x_norm_fp32 = stage_defs[0][1] * torch.rsqrt(stage_defs[1][1] + args.eps)
        stage_defs.append(("x_norm_fp32", x_norm_fp32))
        stage_defs.append(("x_out", y_sg.to(torch.float32)))

        print("\n=== stage compare (dump vs local recompute) ===")
        for stage_name, local_value in stage_defs:
            dump_name = f"{args.stage_prefix}_{stage_name}"
            try:
                stage_path = resolve_unique_dump_file(
                    args.dump_dir,
                    dump_name,
                    index=None,
                    auto_index_policy=args.auto_index_policy,
                )
                dump_value = load_dump_tensor(stage_path).contiguous().to(device)
                if args.verbose:
                    print(f"stage_path[{stage_name}]={stage_path}")
                    print_tensor_meta(f"{dump_name}_dump", dump_value)
                    print_tensor_meta(f"{dump_name}_local", local_value)
                compare_pair(
                    f"{dump_name}_dump_vs_local",
                    dump_value.cpu(),
                    local_value.to(dump_value.dtype).cpu(),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{dump_name}] skip: {exc}")

    hf_eq_sg = (
        hf_sg_max == 0.0 and hf_sg_mean == 0.0 and hf_sg_bit
    )
    hf_eq_dump = (
        hf_dump_max == 0.0 and hf_dump_mean == 0.0 and hf_dump_bit
    )
    sg_eq_dump = (
        sg_dump_max == 0.0 and sg_dump_mean == 0.0 and sg_dump_bit
    )

    print()
    print(f"HF_MATCH_SG_LOCAL: {'YES' if hf_eq_sg else 'NO'}")
    print(f"HF_MATCH_SG_DUMP: {'YES' if hf_eq_dump else 'NO'}")
    print(f"SG_LOCAL_MATCH_SG_DUMP: {'YES' if sg_eq_dump else 'NO'}")

    if hf_eq_sg and hf_eq_dump and sg_eq_dump:
        print("Conclusion: HF RMSNorm and SGLang RMSNorm are fully aligned for this dumped layer0 input.")
    elif sg_eq_dump and not hf_eq_dump:
        print("Conclusion: SGLang local reproduces dump but HF does not; mismatch is in RMSNorm implementation semantics.")
    elif hf_eq_dump and not sg_eq_dump:
        print("Conclusion: HF matches dump but local SGLang setup does not; check SGLang call mode/kwargs.")
    elif not sg_eq_dump:
        print("Conclusion: local SGLang RMSNorm does not yet reproduce dump; check input tensor choice and runtime path.")
    else:
        print("Conclusion: partial mismatch remains; inspect dtype, weight loading, and call path details.")


if __name__ == "__main__":
    main()
