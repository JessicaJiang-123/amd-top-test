#!/usr/bin/env python3
"""
Compare FSDP training-side dumps vs SGLang inference-side dumps
for AMD Triton true-on-policy verification.

Two comparison strategies:

Strategy A — logprob / logit tensors
  Sequential alignment: training pass[k] token[t]  ←→  inference decode step k*N_resp+t
  Cross-validated by printing matching next_token_id pairs.
  NOTE: most reliable when run with --rollout-max-response-len 1 (N_resp=1 → 1-to-1 mapping).

Strategy B — attention intermediate tensors
  Full-prefix comparison: TR[:P] vs SG_prefill[:P], P = min(prefill_len, training_seq_len).
  The prefill pass and the teacher-forcing pass process the same prompt tokens at the
  same positions → should be bit-wise identical under true-on-policy.

Usage:
  Update SG_DIR / TR_DIR below, then:
  PYTHONPATH=/data/true_on_policy/sglang/python \
  python experiment/step2_miles_fsdp_compare/compare_amd_triton.py
"""
from pathlib import Path
import re
import collections

import torch

# ===== Update these per run =====
SG_DIR = Path("/tmp/my_dumps/sglang_dump_1772775991.6409755")   # SGLang inference
TR_DIR = Path("/tmp/my_dumps/sglang_dump_1772776018.0248804")   # FSDP training

DIFF_THRESHOLD = 1e-3

LOGPROB_NAMES = {
    "next_token_logprob_selected",
    "next_token_logits_raw",
    "next_token_logprobs_full",
    "next_token_id",
}

ATTN_INTERMEDIATE_NAMES = {
    "layer0_q_pre_norm", "layer0_k_pre_norm", "layer0_v_pre_norm",
    "layer0_q_post_norm", "layer0_k_post_norm",
    "layer0_q_post_rope", "layer0_k_post_rope",
    "layer0_attn_context_before_o_proj", "layer0_attn_out_after_o_proj",
    "q_pre_norm", "k_pre_norm", "v_pre_norm",
    "q_post_norm", "k_post_norm",
    "q_post_rope", "k_post_rope",
    "attn_context_before_o_proj", "attn_out_last_layer",
}

_PAT = re.compile(
    r"forward_pass_id=(\d+)___rank=(\d+)___name=(.*?)___dump_index=(\d+)\.pt$"
)


def scan(d: Path):
    """Returns {name: [(dump_index, path)]} sorted by dump_index."""
    result = collections.defaultdict(list)
    for p in d.glob("*.pt"):
        m = _PAT.match(p.name)
        if not m:
            continue
        rank, name, idx = int(m.group(2)), m.group(3), int(m.group(4))
        if rank != 0:
            continue
        result[name].append((idx, p))
    for k in result:
        result[k].sort(key=lambda x: x[0])
    return result


def load_pt(p: Path) -> torch.Tensor:
    """Load dump file, preserve original dtype (no implicit .float())."""
    x = torch.load(p, weights_only=False, map_location="cpu")
    if isinstance(x, dict) and "value" in x:
        x = x["value"]
    assert isinstance(x, torch.Tensor), f"expected Tensor, got {type(x)}"
    return x


def report(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    """
    Print dtype, torch.equal, and abs diff.
    Diff is computed in the NATIVE dtype — no silent upcast to float32.
    If dtypes differ, we only report the mismatch (no diff).
    """
    same_dtype = a.dtype == b.dtype
    dtype_str = f"dtype={a.dtype}" if same_dtype else f"dtype={a.dtype}≠{b.dtype}"

    if same_dtype:
        eq = torch.equal(a, b)
        if a.is_floating_point():
            diff = (a - b).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            ok = "✅" if max_abs <= DIFF_THRESHOLD else "❌"
            print(
                f"  {label}  {ok}  bitwise_equal={eq}  "
                f"{dtype_str}  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}"
            )
        else:
            # integer / bool: report mismatch count
            neq = int((a != b).sum().item())
            ok = "✅" if neq == 0 else "❌"
            print(
                f"  {label}  {ok}  bitwise_equal={eq}  "
                f"{dtype_str}  neq_count={neq}/{a.numel()}"
            )
    else:
        print(
            f"  {label}  ⚠️  {dtype_str}  "
            f"shapes={tuple(a.shape)} vs {tuple(b.shape)}  (no diff — fix dtype first)"
        )


# ---------------------------------------------------------------------------
# Structure dump
# ---------------------------------------------------------------------------

def print_dump_structure(sg: dict, tr: dict) -> None:
    common = sorted(set(sg) & set(tr))
    print(f"\n{'Name':<45} {'SG':>4}×  shape×dtype (SG)              {'TR':>4}×  shape×dtype (TR)")
    print("-" * 115)
    for name in common:
        sg_meta = sorted({(tuple(t.shape), str(t.dtype)) for _, p in sg[name] for t in [load_pt(p)]})
        tr_meta = sorted({(tuple(t.shape), str(t.dtype)) for _, p in tr[name] for t in [load_pt(p)]})
        sg_str = "  ".join(f"{s} {d}" for s, d in sg_meta)
        tr_str = "  ".join(f"{s} {d}" for s, d in tr_meta)
        print(
            f"  {name:<43} {len(sg[name]):>4}×  {sg_str:<35}"
            f"  {len(tr[name]):>4}×  {tr_str}"
        )
    print()


# ---------------------------------------------------------------------------
# Strategy A: logprob / logit tensors
# ---------------------------------------------------------------------------

def compare_logprob(name: str, sg_entries, tr_entries, sg_id_entries, tr_id_entries) -> None:
    """
    Sequential alignment:
      training pass[k] row[t]  ←→  inference decode step at flat index k*N_resp+t

    Token-ID cross-check: prints whether next_token_id matches for each pair.
    If IDs don't match, the alignment assumption is broken.
    """
    sg_decode = [(idx, p, load_pt(p)) for idx, p in sg_entries if load_pt(p).shape[0] == 1]
    tr_passes = [(idx, p, load_pt(p)) for idx, p in tr_entries]

    if not sg_decode or not tr_passes:
        print(f"[{name}] SKIP: empty decode steps or training passes")
        return

    n_resp = tr_passes[0][2].shape[0]  # rows in first training pass = N_resp

    # Load token IDs for cross-checking (may be absent)
    sg_decode_ids = (
        [load_pt(p).tolist() for _, p in sg_id_entries if load_pt(p).shape[0] == 1]
        if sg_id_entries else []
    )
    tr_pass_ids = (
        [load_pt(p).tolist() for _, p in tr_id_entries]
        if tr_id_entries else []
    )

    print(f"\n[{name}]  SG decode steps={len(sg_decode)}  TR passes={len(tr_passes)}  N_resp={n_resp}")

    for k, (tr_idx, _tr_path, tr_t) in enumerate(tr_passes):
        for t in range(n_resp):
            flat = k * n_resp + t
            if flat >= len(sg_decode):
                break
            sg_idx, _sg_path, sg_t = sg_decode[flat]

            # Extract the t-th element from training (keep dtype)
            tr_elem = tr_t[t : t + 1] if tr_t.ndim == 1 else tr_t[t : t + 1]
            sg_elem = sg_t  # shape (1, ...) or (1,)

            # Squeeze trailing dim-1 if shapes differ by a leading 1
            if tr_elem.shape != sg_elem.shape:
                try:
                    sg_elem = sg_elem.view_as(tr_elem)
                except RuntimeError:
                    pass

            # Token-ID cross-check
            id_ok = ""
            if sg_decode_ids and tr_pass_ids and name != "next_token_id":
                sg_id_val = sg_decode_ids[flat] if flat < len(sg_decode_ids) else "?"
                tr_id_val = tr_pass_ids[k] if k < len(tr_pass_ids) else "?"
                # tr_id is a list of N_resp ids; pick t-th
                if isinstance(tr_id_val, list) and t < len(tr_id_val):
                    tr_id_val = tr_id_val[t]
                match = "id✅" if sg_id_val == tr_id_val else f"id❌(sg={sg_id_val} tr={tr_id_val})"
                id_ok = f"  [{match}]"

            label = f"pass[{k}] tok[{t}]  tr_idx={tr_idx} sg_idx={sg_idx}{id_ok}"
            report(label, tr_elem, sg_elem)


# ---------------------------------------------------------------------------
# Strategy B: attention intermediate tensors — full prefix comparison
# ---------------------------------------------------------------------------

def compare_attn_intermediate(name: str, sg_entries, tr_entries) -> None:
    """
    Find the first inference PREFILL step (shape[0] > 1).
    Compare TR[:P] vs SG_prefill[:P], P = min(prefill_len, training_seq_len).

    Both sides process the same prompt tokens at the same positions, so these
    should be bit-wise identical under true-on-policy.
    """
    sg_prefill = None
    for idx, p in sg_entries:
        t = load_pt(p)
        if t.ndim >= 2 and t.shape[0] > 1:
            sg_prefill = (idx, p, t)
            break

    if sg_prefill is None:
        print(f"[{name}] SKIP: no prefill found in inference (all decode steps have shape[0]==1)")
        return

    sg_idx, _sg_path, sg_t = sg_prefill
    tr_idx, _tr_path = tr_entries[0]
    tr_t = load_pt(_tr_path)

    P = min(sg_t.shape[0], tr_t.shape[0])
    sg_seg = sg_t[:P]
    tr_seg = tr_t[:P]

    print(
        f"[{name}]  "
        f"SG prefill idx={sg_idx} shape={tuple(sg_t.shape)} dtype={sg_t.dtype}  "
        f"TR first-pass idx={tr_idx} shape={tuple(tr_t.shape)} dtype={tr_t.dtype}  "
        f"comparing [:P={P}]"
    )
    report("prefix", tr_seg, sg_seg)


# ---------------------------------------------------------------------------
# Sanity check: sample alignment vs weight bit-level difference
# ---------------------------------------------------------------------------

SANITY_PRINT_TOKENS = 3   # how many tokens to print
SANITY_PRINT_DIMS  = 8    # how many feature dims to print per token

def sanity_check_sample_alignment(name: str, sg_entries, tr_entries) -> None:
    """
    Print the raw values of the first few tokens (and dims) for both sides.

    Interpretation guide
    --------------------
    - Values completely different  →  different samples are being compared
                                      (alignment bug in Strategy B pairing)
    - Values close but not bit-exact → same sample, tiny weight / precision diff
    - Values bit-exact               → perfect match before any accumulation
    """
    # Find first SG prefill (shape[0] > 1)
    sg_prefill = None
    for idx, p in sg_entries:
        t = load_pt(p)
        if t.ndim >= 2 and t.shape[0] > 1:
            sg_prefill = (idx, p, t)
            break

    if sg_prefill is None:
        print(f"[sanity/{name}] SKIP: no prefill found in SG")
        return

    sg_idx, _sg_path, sg_t = sg_prefill
    tr_idx, _tr_path = tr_entries[0]
    tr_t = load_pt(_tr_path)

    N = min(SANITY_PRINT_TOKENS, sg_t.shape[0], tr_t.shape[0])
    D = min(SANITY_PRINT_DIMS, sg_t.shape[-1], tr_t.shape[-1])

    print()
    print(f"[sanity/{name}]  SG idx={sg_idx} shape={tuple(sg_t.shape)}  TR idx={tr_idx} shape={tuple(tr_t.shape)}")
    print(f"  Printing first {N} tokens × first {D} dims  (to diagnose: sample mismatch vs weight diff)")
    for tok in range(N):
        sg_vals = sg_t[tok, :D].float().tolist()
        tr_vals = tr_t[tok, :D].float().tolist()
        sg_str = "  ".join(f"{v:+.4f}" for v in sg_vals)
        tr_str = "  ".join(f"{v:+.4f}" for v in tr_vals)
        diff   = [(a - b) for a, b in zip(tr_t[tok, :D].float().tolist(), sg_vals)]
        diff_str = "  ".join(f"{v:+.4f}" for v in diff)
        print(f"  tok[{tok}]  SG= {sg_str}")
        print(f"  tok[{tok}]  TR= {tr_str}")
        print(f"  tok[{tok}]  Δ = {diff_str}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"SG_DIR = {SG_DIR}")
    print(f"TR_DIR = {TR_DIR}\n")

    sg = scan(SG_DIR)
    tr = scan(TR_DIR)

    print(f"Inference unique names : {len(sg)}  (total files: {sum(len(v) for v in sg.values())})")
    print(f"Training  unique names : {len(tr)}  (total files: {sum(len(v) for v in tr.values())})")
    common = sorted(set(sg) & set(tr))
    print(f"Common names           : {len(common)}")

    print_dump_structure(sg, tr)

    # ---- Strategy A ----
    print("=" * 60)
    print("Strategy A: logprob / logit / id comparison")
    print("  (sequential alignment; cross-checked by next_token_id)")
    print("=" * 60)
    sg_ids = sg.get("next_token_id", [])
    tr_ids = tr.get("next_token_id", [])
    for name in sorted(LOGPROB_NAMES & set(common)):
        compare_logprob(name, sg[name], tr[name], sg_ids, tr_ids)

    # ---- Strategy B ----
    print()
    print("=" * 60)
    print("Strategy B: attention intermediates, full-prefix comparison")
    print("  (TR[:P] vs SG_prefill[:P], P = min(prefill_len, seq_len))")
    print("=" * 60)

    # Sanity check: print actual values of first few tokens for layer0_q_pre_norm
    # to determine whether sample mismatch or weight bit-level difference
    sanity_name = "layer0_q_pre_norm"
    if sanity_name in sg and sanity_name in tr:
        sanity_check_sample_alignment(sanity_name, sg[sanity_name], tr[sanity_name])

    for name in sorted(ATTN_INTERMEDIATE_NAMES & set(common)):
        compare_attn_intermediate(name, sg[name], tr[name])


if __name__ == "__main__":
    import datetime
    import sys

    RESULTS_DIR = Path(__file__).parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sg_tag = SG_DIR.name.replace("sglang_dump_", "sg")
    tr_tag = TR_DIR.name.replace("sglang_dump_", "tr")
    out_path = RESULTS_DIR / f"{timestamp}__{sg_tag}__{tr_tag}.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(out_path, "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__

    print(f"\nResults saved to: {out_path}")
