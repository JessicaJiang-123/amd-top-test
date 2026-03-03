#!/usr/bin/env python3
from pathlib import Path

import torch

# ====== 按你的当前实验修改这两个目录 ======
HF_DIR = Path("/tmp/dumper/sglang_dump_1772426302.4763207")
SG_DIR = Path("/tmp/dumper/sglang_dump_1772426267.912329")

# ====== 按你的当前实验修改这三个 index 映射 ======
# 例子：
#   HF_INDEX = {"attn_out_last_layer": 5, "final_hidden_before_lm_head": 6, "lm_head_weight": 7}
#   SG_INDEX = {"attn_out_last_layer": 31, "final_hidden_before_lm_head": 32, "lm_head_weight": 33}
HF_INDEX = {
    "attn_out_last_layer": 1,
    "final_hidden_before_lm_head": 2,
    "lm_head_weight": 3,
}
SG_INDEX = {
    "attn_out_last_layer": 1,
    "final_hidden_before_lm_head": 2,
    "lm_head_weight": 3,
}

# True: 将 hidden 类张量统一对齐到“单步语义”：
# - [B, T, H] -> [B, H]（取最后一步）
# - [T, H] -> [1, H]（取最后一个 token）
ALIGN_TO_SINGLE_STEP = True


def load_value(d: Path, name: str, idx: int) -> torch.Tensor:
    p = d / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    obj = torch.load(p, weights_only=False, map_location="cpu")
    x = obj["value"] if isinstance(obj, dict) and "value" in obj else obj
    return x


def align_single_step(name: str, x: torch.Tensor) -> torch.Tensor:
    if not ALIGN_TO_SINGLE_STEP:
        return x

    if name in ("attn_out_last_layer", "final_hidden_before_lm_head"):
        # 常见1：teacher-forcing => [B, T, H]
        if x.ndim == 3:
            return x[:, -1, :]
        # 常见2：single-step prefill path => [T, H]
        if x.ndim == 2 and x.shape[0] > 1:
            return x[-1:, :]
    return x


def compare(name: str) -> None:
    x_hf = load_value(HF_DIR, name, HF_INDEX[name])
    x_sg = load_value(SG_DIR, name, SG_INDEX[name])
    x_hf = align_single_step(name, x_hf)
    x_sg = align_single_step(name, x_sg)

    print(f"\n[{name}]")
    print("  hf shape/dtype:", tuple(x_hf.shape), x_hf.dtype)
    print("  sg shape/dtype:", tuple(x_sg.shape), x_sg.dtype)

    if x_hf.shape != x_sg.shape:
        print("  -> shape mismatch, skip")
        return

    bitwise = torch.equal(x_hf, x_sg)
    print("  torch.equal:", bitwise)

    if x_hf.dtype.is_floating_point or x_sg.dtype.is_floating_point:
        diff = (x_hf.float() - x_sg.float()).abs()
        print("  max_abs:", diff.max().item())
        print("  mean_abs:", diff.mean().item())
    else:
        neq = (x_hf != x_sg).sum().item()
        print("  neq_cnt:", neq)

    if x_hf.numel() == 1:
        print("  hf value:", x_hf.item())
        print("  sg value:", x_sg.item())


def main() -> None:
    print("HF_DIR =", HF_DIR)
    print("SG_DIR =", SG_DIR)
    print("ALIGN_TO_SINGLE_STEP =", ALIGN_TO_SINGLE_STEP)

    for name in [
        "attn_out_last_layer",
        "final_hidden_before_lm_head",
        "lm_head_weight",
    ]:
        compare(name)


if __name__ == "__main__":
    main()
