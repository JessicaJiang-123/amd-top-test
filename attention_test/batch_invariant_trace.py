"""
Batch-invariant ops 调用计数追踪。

在 enable_batch_invariant_mode() 之前 monkey-patch，注册到 aten 的即为带计数的包装函数。
不修改 sglang 源码，所有逻辑在 /app/program 内。
"""
import os
from typing import Dict, Iterable, Optional, Set

import torch

VALID_BATCH_INVARIANT_OPS = {"mm", "addmm", "bmm", "mean", "log_softmax"}

# mm 追踪：记录每次调用的元数据，首次 NaN 时保存张量
_MM_TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_mm_trace")
_MM_CALL_IDX = [0]
_FIRST_NAN_SAVED = [False]

# 全局计数
_call_counts: Dict[str, int] = {
    "mm": 0,
    "addmm": 0,
    "bmm": 0,
    "log_softmax": 0,
    "mean": 0,
}


def _make_traced(name: str, fn):
    def traced(*args, **kwargs):
        _call_counts[name] += 1
        return fn(*args, **kwargs)

    return traced


def _make_mm_traced_with_nan_debug(fn):
    """包装 mm：记录每次 (idx, shape, out_stats)，首次 NaN 时保存 a,b"""

    def traced(a, b):
        idx = _MM_CALL_IDX[0]
        _MM_CALL_IDX[0] += 1

        out = fn(a, b)

        M, K = a.shape
        _, N = b.shape
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()

        if has_nan or has_inf:
            out_min = out_max = out_mean = float("nan")
        else:
            out_f = out.float()
            out_min = out_f.min().item()
            out_max = out_f.max().item()
            out_mean = out_f.mean().item()

        os.makedirs(_MM_TRACE_DIR, exist_ok=True)
        trace_path = os.path.join(_MM_TRACE_DIR, "mm_trace.txt")
        with open(trace_path, "a") as f:
            status = "NaN" if has_nan else ("INF" if has_inf else "ok")
            f.write(
                f"Step {idx:4d}: ({M:5d},{K:5d}) @ ({K:5d},{N:5d})  "
                f"out=[{out_min:10.4f},{out_max:10.4f}] mean={out_mean:10.4f}  {status}\n"
            )

        if has_nan and not _FIRST_NAN_SAVED[0]:
            _FIRST_NAN_SAVED[0] = True
            save_path = os.path.join(_MM_TRACE_DIR, "first_nan_step.pt")
            torch.save(
                {"step": idx, "a": a.clone(), "b": b.clone(), "out": out.clone()},
                save_path,
            )
            print(f"[DEBUG] 首次 NaN 在 step {idx}，已保存到 {save_path}")

        return out

    return traced


def reset_mm_trace():
    """清空 mm 追踪状态和文件，供新一次运行使用"""
    _MM_CALL_IDX[0] = 0
    _FIRST_NAN_SAVED[0] = False
    trace_path = os.path.join(_MM_TRACE_DIR, "mm_trace.txt")
    if os.path.exists(trace_path):
        os.remove(trace_path)


def enable_batch_invariant_mode_with_tracing(
    ops: Optional[Iterable[str]] = None,
    debug_mm_trace: bool = False,
):
    """
    先 patch 再 enable，使注册到 aten 的为带计数的版本。

    Args:
        ops: 要启用的 op 列表，如 ["mm", "mean", "bmm", "addmm", "log_softmax"]。
             未列出的 op 不启用（可逐个测试排查）。若为 None 则启用全部。
        debug_mm_trace: 若 True 且启用 mm，记录每次 mm 的 shape/out_stats 到 mm_trace.txt，
                       首次 NaN 时保存 a,b 到 first_nan_step.pt。
    """
    # 导入实际包含 impl 的模块（batch_invariant_ops.py）
    bio = __import__(
        "sglang.srt.batch_invariant_ops.batch_invariant_ops",
        fromlist=["mm_batch_invariant", "addmm_batch_invariant", "bmm_batch_invariant",
                  "_log_softmax_batch_invariant", "mean_batch_invariant", "enable_batch_invariant_mode"],
    )

    enable_mm = enable_addmm = enable_mean = enable_bmm = enable_log_softmax = True
    if ops is not None:
        op_set: Set[str] = {s.strip().lower() for s in ops if s}
        invalid = op_set - VALID_BATCH_INVARIANT_OPS
        if invalid:
            raise ValueError(
                f"未知的 batch-invariant op: {invalid}，有效值为 {VALID_BATCH_INVARIANT_OPS}"
            )
        enable_mm = "mm" in op_set
        enable_addmm = "addmm" in op_set
        enable_mean = "mean" in op_set
        enable_bmm = "bmm" in op_set
        enable_log_softmax = "log_softmax" in op_set

    # 在 enable 之前替换为带计数的版本（仅对要启用的 op 做 trace）
    if enable_mm:
        if debug_mm_trace:
            bio.mm_batch_invariant = _make_mm_traced_with_nan_debug(bio.mm_batch_invariant)
        bio.mm_batch_invariant = _make_traced("mm", bio.mm_batch_invariant)
    if enable_addmm:
        bio.addmm_batch_invariant = _make_traced("addmm", bio.addmm_batch_invariant)
    if enable_bmm:
        bio.bmm_batch_invariant = _make_traced("bmm", bio.bmm_batch_invariant)
    if enable_log_softmax:
        bio._log_softmax_batch_invariant = _make_traced("log_softmax", bio._log_softmax_batch_invariant)
    if enable_mean:
        bio.mean_batch_invariant = _make_traced("mean", bio.mean_batch_invariant)

    # Miles true-on-policy 使用 enable_bmm=False：Qwen3 RoPE 的 bmm 不替换才能对齐
    bio.enable_batch_invariant_mode(enable_bmm=False)


def get_call_counts() -> Dict[str, int]:
    return dict(_call_counts)


def reset_call_counts():
    for k in _call_counts:
        _call_counts[k] = 0


def print_call_counts():
    print("[batch_invariant_ops] 调用统计:", _call_counts)