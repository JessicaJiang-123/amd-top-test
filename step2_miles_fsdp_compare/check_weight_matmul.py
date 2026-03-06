#!/usr/bin/env python3
"""
Verify whether the 1-2 ULP difference in layer0_q_pre_norm comes from:
  (A) Weight loading difference  -- SGLang merged QKV vs FSDP separate q_proj
  (B) matmul SHAPE difference    -- merged [seq, hidden]@[hidden, q+k+v] vs
                                     separate [seq, hidden]@[hidden, q]
  (C) both

Two experiments are run:

  Exp 1 — Weight check
    Load HF checkpoint, extract Q weight two ways:
      • qkv_weight[:q_dim, :]  (how SGLang stores it after merge)
      • q_proj_weight           (how FSDP/HF loads it directly)
    Confirm they are bit-identical.

  Exp 2 — matmul shape effect
    With bit-identical Q weight and a random BF16 input x:
      • merged  matmul: (x @ qkv_weight.T)[:, :q_dim]
      • separate matmul: x @ q_weight.T
    Use matmul_persistent (batch_invariant Triton kernel) for both.
    Report whether the Q output differs.

Usage (inside the training container):
  PYTHONPATH=/data/true_on_policy/sglang/python:/data/true_on_policy/miles \
  python experiment/step2_miles_fsdp_compare/check_weight_matmul.py \
    --model-path /root/models/Qwen3-0.6B
"""
import argparse
import sys
import torch

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the HF checkpoint (e.g. /root/models/Qwen3-0.6B)")
parser.add_argument("--seq-len-sg", type=int, default=66,
                    help="Prefill sequence length used by SGLang (default: 66)")
parser.add_argument("--seq-len-tr", type=int, default=128,
                    help="Sequence length used by FSDP training (default: 128)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load HF checkpoint weights (layer 0 only)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Model: {args.model_path}")
print(f"{'='*60}\n")

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
print(f"hidden_size={cfg.hidden_size}  "
      f"num_heads={cfg.num_attention_heads}  "
      f"num_kv_heads={cfg.num_key_value_heads}  "
      f"head_dim={getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}")

head_dim  = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
q_dim     = cfg.num_attention_heads * head_dim
k_dim     = cfg.num_key_value_heads * head_dim
v_dim     = cfg.num_key_value_heads * head_dim
hidden    = cfg.hidden_size
print(f"q_dim={q_dim}  k_dim={k_dim}  v_dim={v_dim}  hidden={hidden}\n")

# Load safetensors / bin weights for layer 0
import os
from pathlib import Path
model_path = Path(args.model_path)

# Try safetensors first
weight_files = list(model_path.glob("*.safetensors"))
if weight_files:
    from safetensors.torch import load_file
    state_dict = {}
    for wf in sorted(weight_files):
        state_dict.update(load_file(str(wf), device="cpu"))
else:
    import glob
    weight_files = list(model_path.glob("pytorch_model*.bin"))
    state_dict = {}
    for wf in sorted(weight_files):
        state_dict.update(torch.load(str(wf), map_location="cpu", weights_only=True))

q_weight = state_dict["model.layers.0.self_attn.q_proj.weight"]  # [q_dim, hidden]
k_weight = state_dict["model.layers.0.self_attn.k_proj.weight"]  # [k_dim, hidden]
v_weight = state_dict["model.layers.0.self_attn.v_proj.weight"]  # [v_dim, hidden]
print(f"Loaded weights:  q={tuple(q_weight.shape)} {q_weight.dtype}  "
      f"k={tuple(k_weight.shape)}  v={tuple(v_weight.shape)}")

# ---------------------------------------------------------------------------
# Experiment 1: weight identity check
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Experiment 1: Weight identity  (SGLang qkv_merged[:q_dim] vs HF q_proj)")
print(f"{'='*60}")

# SGLang merges [q, k, v] along dim-0 → qkv_weight[0:q_dim] should equal q_weight
qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [q+k+v, hidden]
qkv_q_slice = qkv_weight[:q_dim, :]  # Q portion

bits_identical = torch.equal(qkv_q_slice, q_weight)
if bits_identical:
    print(f"  ✅  qkv_weight[:q_dim] == q_proj.weight  (bit-identical)")
else:
    diff = (qkv_q_slice.float() - q_weight.float()).abs()
    print(f"  ❌  qkv_weight[:q_dim] != q_proj.weight  "
          f"max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}")

# ---------------------------------------------------------------------------
# Experiment 2: matmul shape effect with batch_invariant Triton kernel
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Experiment 2: matmul shape effect")
print(f"  SG: x[{args.seq_len_sg}, {hidden}] @ qkv_weight.T[{hidden}, {q_dim+k_dim+v_dim}]  → take [:, :q_dim]")
print(f"  TR: x[{args.seq_len_tr}, {hidden}] @ q_weight.T[{hidden}, {q_dim}]  → take [:min_seq]")
print(f"  Compare overlapping prefix rows  (first min({args.seq_len_sg},{args.seq_len_tr}) rows)")
print(f"{'='*60}")

# Enable batch_invariant Triton kernel (mirrors what both sides do at runtime)
try:
    sys.path.insert(0, str(Path(__file__).parents[3] / "sglang/python"))
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        matmul_persistent,
        enable_batch_invariant_mode,
    )
    # patch aten::mm so F.linear also goes through it
    enable_batch_invariant_mode(enable_bmm=False)
    HAS_TRITON = True
    print("  batch_invariant Triton kernel: ✅ loaded\n")
except Exception as e:
    HAS_TRITON = False
    print(f"  batch_invariant Triton kernel: ❌ not available ({e})\n"
          f"  → falling back to torch.mm (results may differ from runtime)\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("  ⚠️  No CUDA device — using CPU matmul (Triton unavailable on CPU, results approximate)\n")

torch.manual_seed(args.seed)

# Same random input, two different lengths
P = min(args.seq_len_sg, args.seq_len_tr)

# SG side: seq_len_sg rows, merged QKV matmul
x_sg = torch.randn(args.seq_len_sg, hidden, dtype=torch.bfloat16, device=device)
W_qkv = qkv_weight.to(device=device)   # [q+k+v, hidden]
W_q   = q_weight.to(device=device)     # [q_dim, hidden]

# TR side: seq_len_tr rows, same first P rows as x_sg
x_tr = torch.zeros(args.seq_len_tr, hidden, dtype=torch.bfloat16, device=device)
x_tr[:args.seq_len_sg] = x_sg          # first seq_len_sg rows are identical

def do_matmul(x, W):
    """x @ W.T  — goes through batch_invariant mm if enabled."""
    if device == "cpu":
        return torch.mm(x, W.t())
    return torch.mm(x, W.t())          # aten::mm is patched if batch_invariant active

# SG: merged QKV, take Q slice
out_qkv = do_matmul(x_sg, W_qkv)      # [seq_len_sg, q+k+v]
out_sg_q = out_qkv[:P, :q_dim]        # [P, q_dim]

# TR: separate Q only
out_q    = do_matmul(x_tr, W_q)       # [seq_len_tr, q_dim]
out_tr_q = out_q[:P, :]               # [P, q_dim]

eq = torch.equal(out_sg_q, out_tr_q)
diff = (out_sg_q.float() - out_tr_q.float()).abs()
max_abs  = diff.max().item()
mean_abs = diff.mean().item()

ok = "✅" if eq else "❌"
print(f"  {ok}  bit_identical={eq}  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}")

if not eq:
    print(f"\n  First 3 rows × first 8 dims:")
    for t in range(min(3, P)):
        sg_vals  = out_sg_q[t, :8].tolist()
        tr_vals  = out_tr_q[t, :8].tolist()
        delta    = [(a - b) for a, b in zip(tr_vals, sg_vals)]
        print(f"    tok[{t}]  SG= " + "  ".join(f"{v:+.4f}" for v in sg_vals))
        print(f"    tok[{t}]  TR= " + "  ".join(f"{v:+.4f}" for v in tr_vals))
        print(f"    tok[{t}]  Δ = " + "  ".join(f"{v:+.4f}" for v in delta))

# ---------------------------------------------------------------------------
# Experiment 3: same shape, different seq_len (isolate seq_len effect)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Experiment 3: Same Q weight, same input, different seq_len")
print(f"  SG: x[{args.seq_len_sg}, {hidden}] @ q_weight.T  (q-only, sg seq_len)")
print(f"  TR: x[{args.seq_len_tr}, {hidden}] @ q_weight.T  (q-only, tr seq_len)")
print(f"  → isolates seq_len tiling effect from merged-QKV effect")
print(f"{'='*60}")

out_sg_qonly = do_matmul(x_sg, W_q)[:P]             # [P, q_dim]
out_tr_qonly = do_matmul(x_tr, W_q)[:P]             # [P, q_dim]

eq3 = torch.equal(out_sg_qonly, out_tr_qonly)
diff3 = (out_sg_qonly.float() - out_tr_qonly.float()).abs()
ok3 = "✅" if eq3 else "❌"
print(f"  {ok3}  bit_identical={eq3}  max_abs={diff3.max().item():.3e}  mean_abs={diff3.mean().item():.3e}")

# ---------------------------------------------------------------------------
# Experiment 4: RMSNorm weight-multiply order  (SGLang forward_native vs HF Qwen3RMSNorm)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Experiment 4: RMSNorm weight-multiply order")
print("  SGLang forward_native: out = (fp32_x * bf16_weight).to(bf16)")
print("  HF Qwen3RMSNorm:       out = bf16_weight * fp32_x.to(bf16)")
print(f"{'='*60}")

rms_norm_eps = getattr(cfg, "rms_norm_eps", 1e-6)

# Random bf16 input
torch.manual_seed(args.seed + 1)
x_norm_in = torch.randn(args.seq_len_sg, hidden, dtype=torch.bfloat16, device=device)

# Same RMSNorm weight (use embed weight shape for simplicity)
w_norm = torch.randn(hidden, dtype=torch.bfloat16, device=device)

# Norm weight in fp32 (SGLang stores it as fp32 via weight_dtype=torch.float32)
w_norm_fp32 = w_norm.to(torch.float32)  # same values, widened to fp32

def rmsnorm_sglang(x, w_fp32, eps):
    """SGLang forward_native with rl_on_policy_target params:
       weight_dtype=float32, cast_x_before_out_mul=True, override_orig_dtype=float32
       → entire computation in fp32, output is fp32"""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    # cast_x_before_out_mul=True, orig_dtype=float32 → no-op cast, fp32 * fp32
    return w_fp32 * x_normed   # fp32 output

def rmsnorm_hf(x, w_bf16, eps):
    """HF Qwen3RMSNorm:
       weight is bf16, normalize in fp32, cast normed_x back to bf16 before multiply
       → output is bf16"""
    input_dtype = x.dtype   # bf16
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return w_bf16 * x_normed.to(input_dtype)  # bf16 * bf16 → bf16 output

out_sg_fp32 = rmsnorm_sglang(x_norm_in, w_norm_fp32, rms_norm_eps)   # fp32
out_hf_bf16 = rmsnorm_hf(x_norm_in, w_norm, rms_norm_eps)             # bf16

print(f"  SG norm output dtype: {out_sg_fp32.dtype}  (fp32, due to override_orig_dtype=float32)")
print(f"  HF norm output dtype: {out_hf_bf16.dtype}  (bf16, standard Qwen3RMSNorm)")

# Compare norms (must cast SG to bf16 to compare on equal footing)
out_sg_cast = out_sg_fp32.to(torch.bfloat16)
eq4 = torch.equal(out_sg_cast, out_hf_bf16)
diff4 = (out_sg_cast.float() - out_hf_bf16.float()).abs()
ok4 = "✅" if eq4 else "❌"
print(f"\n  Norm outputs (SG.to(bf16) vs HF.bf16):")
print(f"  {ok4}  bit_identical={eq4}  max_abs={diff4.max().item():.3e}  mean_abs={diff4.mean().item():.3e}")

# Then pass BOTH norm outputs through the SAME q_proj matmul
# to see whether the dtype difference in norm output causes q_pre_norm to differ
W_q_dev    = W_q.to(device=device)              # bf16
W_q_fp32   = W_q_dev.float()                    # promoted to fp32 (F.linear type-promotion rule)

# SG: fp32 norm output → F.linear promotes bf16_weight to fp32 → mm(fp32, fp32) → fp32 → .to(bf16)
q_sg = torch.mm(out_sg_fp32, W_q_fp32.t()).to(torch.bfloat16)

# HF: bf16 norm output → F.linear same dtype → mm(bf16, bf16) → bf16
q_hf = torch.mm(out_hf_bf16, W_q_dev.t())  # bf16 × bf16 → bf16

eq4b = torch.equal(q_sg, q_hf)
diff4b = (q_sg.float() - q_hf.float()).abs()
ok4b = "✅" if eq4b else "❌"
print(f"\n  q_proj outputs after norm (SG fp32→matmul→bf16  vs  HF bf16→matmul→bf16):")
print(f"  {ok4b}  bit_identical={eq4b}  max_abs={diff4b.max().item():.3e}  mean_abs={diff4b.mean().item():.3e}")

if not eq4b:
    print(f"\n  First 3 rows × first 8 dims:")
    for t in range(min(3, args.seq_len_sg)):
        sg_vals = q_sg[t, :8].tolist()
        hf_vals = q_hf[t, :8].tolist()
        delta   = [(a - b) for a, b in zip(sg_vals, hf_vals)]
        print(f"    tok[{t}]  SG= " + "  ".join(f"{v:+.4f}" for v in sg_vals))
        print(f"    tok[{t}]  HF= " + "  ".join(f"{v:+.4f}" for v in hf_vals))
        print(f"    tok[{t}]  Δ = " + "  ".join(f"{v:+.4f}" for v in delta))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"  Exp 1  Weight identity (qkv_merged[:q] vs q_proj):  {'✅ identical' if bits_identical else '❌ DIFFERENT'}")
print(f"  Exp 2  matmul shape  (merged QKV seq={args.seq_len_sg} vs q-only seq={args.seq_len_tr}):  {'✅ identical' if eq else f'❌ DIFFERENT  max_abs={max_abs:.3e}'}")
print(f"  Exp 3  seq_len only  (q-only seq={args.seq_len_sg} vs q-only seq={args.seq_len_tr}):  {'✅ identical' if eq3 else f'❌ DIFFERENT  max_abs={diff3.max().item():.3e}'}")
print(f"  Exp 4a RMSNorm output (SG fp32 vs HF bf16 cast to bf16):  {'✅ identical' if eq4 else f'❌ DIFFERENT  max_abs={diff4.max().item():.3e}'}")
print(f"  Exp 4b q_proj output  (SG: fp32_norm→mm→bf16  vs  HF: bf16_norm→mm→bf16):  {'✅ identical' if eq4b else f'❌ DIFFERENT  max_abs={diff4b.max().item():.3e}'}")
print()
if not eq4b:
    print("  → Root cause confirmed: SGLang input_layernorm outputs fp32 (override_orig_dtype=float32)")
    print("    while HF Qwen3RMSNorm outputs bf16.")
    print("    The downstream q_proj matmul computes fp32_input × bf16_weight on SGLang side")
    print("    vs bf16_input × bf16_weight on FSDP side, giving different bf16 results.")
    print()
    print("  Fix: replace HF Qwen3RMSNorm in the FSDP model with SGLang's RMSNorm using the same")
    print("    norm_kwargs (weight_dtype=float32, cast_x_before_out_mul=True, override_orig_dtype=float32)")
    print("    so both sides produce fp32 norm output and pass fp32 to q_proj.")
elif not eq and eq3:
    print("  → Root cause: merged QKV matmul shape causes different tiling.")
elif not eq and not eq3:
    print("  → Root cause: BOTH merged-QKV shape AND seq_len tiling contribute.")
elif eq and not eq3:
    print("  → Root cause: seq_len difference causes different tiling in Triton kernel.")
else:
    print("  → All experiments bit-identical. Difference must come from weight sync or other runtime factor.")
