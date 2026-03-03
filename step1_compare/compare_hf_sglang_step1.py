#!/usr/bin/env python3
from pathlib import Path
import torch

# ====== 按你的当前实验修改这两个目录 ======
HF_DIR = Path("/tmp/dumper/sglang_dump_1772413021.0551426")   # HF侧那次dump目录
SG_DIR = Path("/tmp/dumper/sglang_dump_1772412949.2573981")  # SGLang侧dump目录

# 你指定的对应关系
HF_INDEX = {
    "next_token_logits_raw": 1,
    "next_token_logprobs_full": 2,
    "next_token_id": 3,
    "next_token_logprob_selected": 4,
}
SG_INDEX = {
    "next_token_logits_raw": 19,
    "next_token_id": 20,
    "next_token_logprobs_full": 21,
    "next_token_logprob_selected": 22,
}

def load_value(d: Path, name: str, idx: int) -> torch.Tensor:
    p = d / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    obj = torch.load(p, weights_only=False)
    x = obj["value"] if isinstance(obj, dict) and "value" in obj else obj
    return x

def align_hf_to_single_step(name: str, x: torch.Tensor) -> torch.Tensor:
    # HF是全序列teacher-forcing，取最后一步与SGLang单步对齐
    if name in ("next_token_logits_raw", "next_token_logprobs_full"):
        # [B, T, V] -> [B, V]
        if x.ndim == 3:
            return x[:, -1, :]
    if name in ("next_token_id", "next_token_logprob_selected"):
        # [B, T] -> [B]
        if x.ndim == 2:
            return x[:, -1]
    return x

def compare(name: str):
    x_hf = load_value(HF_DIR, name, HF_INDEX[name])
    x_sg = load_value(SG_DIR, name, SG_INDEX[name])

    x_hf = align_hf_to_single_step(name, x_hf)

    print(f"\n[{name}]")
    print("  hf shape/dtype:", tuple(x_hf.shape), x_hf.dtype)
    print("  sg shape/dtype:", tuple(x_sg.shape), x_sg.dtype)

    if x_hf.shape != x_sg.shape:
        print("  -> shape mismatch, skip")
        return

    bitwise = torch.equal(x_hf, x_sg)
    print("  torch.equal:", bitwise)

    if x_hf.dtype.is_floating_point:
        diff = (x_hf.float() - x_sg.float()).abs()
        print("  max_abs:", diff.max().item())
        print("  mean_abs:", diff.mean().item())
    else:
        neq = (x_hf != x_sg).sum().item()
        print("  neq_cnt:", neq)

    # 方便看单值
    if x_hf.numel() == 1:
        print("  hf value:", x_hf.item())
        print("  sg value:", x_sg.item())

def main():
    print("HF_DIR =", HF_DIR)
    print("SG_DIR =", SG_DIR)

    for n in [
        "next_token_logits_raw",
        "next_token_logprobs_full",
        "next_token_id",
        "next_token_logprob_selected",
    ]:
        compare(n)

if __name__ == "__main__":
    main()