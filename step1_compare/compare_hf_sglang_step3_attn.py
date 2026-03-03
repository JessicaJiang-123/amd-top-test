#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


DEFAULT_HF_DIR = Path("/tmp/dumper/sglang_dump_1772518691.2323422")
DEFAULT_SG_DIR = Path("/tmp/dumper/sglang_dump_1772518041.371588")

DEFAULT_HF_INDEX = {
    "layer0_hidden_in": 6,
    "layer0_attn_out": 7,
    "layer0_block_out": 8,
    "input_ids_for_compare": 1,
    "embedding_output": 2,
    "layer0_positions": 3,
    "layer0_attn_input_raw": 4,
    "layer0_attn_after_input_layernorm_only": 5,
    "layer0_attn_input_after_prepare": 5,
    "attn_input_last_layer": 9,
    "q_pre_norm": 10,
    "k_pre_norm": 11,
    "v_pre_norm": 12,
    "q_post_norm": -1,
    "k_post_norm": -1,
    "q_post_rope": -1,
    "k_post_rope": -1,
    "attn_context_before_o_proj": 13,
    "attn_out_last_layer": 14,
    "final_hidden_before_lm_head": 15,
    "lm_head_weight": 16,
}

DEFAULT_SG_INDEX = {
    "layer0_hidden_in": 64,
    "layer0_q_pre_norm": 65,
    "layer0_k_pre_norm": 66,
    "layer0_v_pre_norm": 67,
    "layer0_attn_context_before_o_proj": 68,
    "layer0_attn_out": 69,
    "layer0_block_out": 70,
    "input_ids_for_compare": 58,
    "embedding_output": 59,
    "layer0_attn_input_raw": 60,
    "layer0_positions": 61,
    "layer0_attn_after_input_layernorm_only": 62,
    "layer0_attn_input_after_prepare": 63,
    "attn_input_last_layer": 71,
    "q_pre_norm": 72,
    "k_pre_norm": 73,
    "v_pre_norm": 74,
    "q_post_norm": 75,
    "k_post_norm": 76,
    "q_post_rope": 77,
    "k_post_rope": 78,
    "attn_context_before_o_proj": 79,
    "attn_out_last_layer": 80,
    "final_hidden_before_lm_head": 81,
    "lm_head_weight": -1,
}

HF_NAME_OVERRIDE = {
    "layer0_attn_after_input_layernorm_only": "layer0_attn_input_after_prepare",
}

SG_NAME_OVERRIDE = {
    "layer0_q_pre_norm": "layer0_q_pre_norm",
    "layer0_k_pre_norm": "layer0_k_pre_norm",
    "layer0_v_pre_norm": "layer0_v_pre_norm",
    "layer0_attn_context_before_o_proj": "layer0_attn_context_before_o_proj",
}

ALL_COMPARE_NAMES = [
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
    "layer0_attn_input_raw",
    "layer0_attn_after_input_layernorm_only",
    "layer0_attn_input_after_prepare",
    "layer0_hidden_in",
    "layer0_attn_out",
    "layer0_block_out",
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
    "lm_head_weight",
]

FOCUS_TO_NAMES = {
    "full": ALL_COMPARE_NAMES,
    "layer0": [
        "input_ids_for_compare",
        "embedding_output",
        "layer0_positions",
        "layer0_attn_input_raw",
        "layer0_attn_after_input_layernorm_only",
        "layer0_attn_input_after_prepare",
        "layer0_hidden_in",
        "layer0_attn_out",
        "layer0_block_out",
    ],
    "last_layer": [
        "attn_input_last_layer",
        "q_pre_norm",
        "k_pre_norm",
        "v_pre_norm",
        "q_post_norm",
        "k_post_norm",
        "q_post_rope",
        "k_post_rope",
        "attn_context_before_o_proj",
        "attn_out_last_layer",
        "final_hidden_before_lm_head",
        "lm_head_weight",
    ],
}

SECTION_TITLES = {
    "input_ids_for_compare": "Inputs",
    "layer0_attn_input_raw": "Layer 0 Prepare",
    "layer0_hidden_in": "Layer 0 Attention",
    "attn_input_last_layer": "Last Layer Attention",
    "final_hidden_before_lm_head": "LM Head",
}

ALIGN_TO_SINGLE_STEP = True

ALIGN_NAMES = {
    "layer0_hidden_in",
    "layer0_attn_out",
    "layer0_block_out",
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
}

HF_DROP_LAST_TOKEN_NAMES = {
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
}

SQUEEZE_BATCH1_NAMES = {
    "layer0_hidden_in",
    "layer0_attn_out",
    "layer0_block_out",
    "input_ids_for_compare",
    "embedding_output",
    "layer0_positions",
    "layer0_attn_input_raw",
    "layer0_attn_after_input_layernorm_only",
    "layer0_attn_input_after_prepare",
    "attn_input_last_layer",
    "q_pre_norm",
    "k_pre_norm",
    "v_pre_norm",
    "q_post_norm",
    "k_post_norm",
    "q_post_rope",
    "k_post_rope",
    "attn_context_before_o_proj",
    "attn_out_last_layer",
    "final_hidden_before_lm_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF and SGLang debug dumps for deterministic alignment."
    )
    parser.add_argument("--hf-dir", type=Path, default=DEFAULT_HF_DIR)
    parser.add_argument("--sg-dir", type=Path, default=DEFAULT_SG_DIR)
    parser.add_argument("--hf-index-json", type=Path, default=None)
    parser.add_argument("--sg-index-json", type=Path, default=None)
    parser.add_argument("--auto-index", action="store_true", default=True)
    parser.add_argument(
        "--focus",
        choices=sorted(FOCUS_TO_NAMES.keys()),
        default="full",
    )
    parser.add_argument("--output-txt", type=Path, default=None)
    return parser.parse_args()


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, line: str = "") -> None:
        self.lines.append(line)
        print(line)

    def flush_to(self, path: Path | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _extract_dump_index(p: Path) -> int:
    return int(p.stem.split("___dump_index=")[1])


def _score_dump_candidate(name: str, value: torch.Tensor) -> int:
    shape = tuple(value.shape)
    ndim = len(shape)

    if name == "lm_head_weight":
        return 100 if ndim == 2 and shape[0] > 1 and shape[1] > 1 else -1

    if name in {"input_ids_for_compare", "layer0_positions"}:
        if ndim == 1:
            return 100 if shape[0] > 1 else 10
        if ndim == 2:
            if shape[1] > 1:
                return 90
            if shape[0] > 1:
                return 80
            return 10
        return -1

    if ndim == 3:
        if shape[1] > 1 and shape[2] > 1:
            return 100
        if shape[2] > 1:
            return 40
        return -1
    if ndim == 2:
        if shape[0] > 1 and shape[1] > 1:
            return 95
        if shape[1] > 1:
            return 50
        return -1
    if ndim == 1:
        return 0 if shape[0] > 1 else -1
    return -1


def _load_index_map(path: Path | None, default_map: dict[str, int]) -> dict[str, int]:
    if path is None:
        return dict(default_map)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {k: int(v) for k, v in payload.items()}


def _discover_index_map(
    dump_dir: Path,
    compare_names: list[str],
    side: str,
    fallback_map: dict[str, int],
) -> dict[str, int]:
    resolved = dict(fallback_map)
    for name in compare_names:
        if side == "hf":
            dump_name = HF_NAME_OVERRIDE.get(name, name)
        else:
            dump_name = SG_NAME_OVERRIDE.get(name, name)
        matches = sorted(
            dump_dir.glob(f"forward_pass_id=0___rank=0___name={dump_name}___dump_index=*.pt"),
            key=_extract_dump_index,
        )
        if not matches:
            if name not in resolved:
                resolved[name] = -1
            continue

        best_path = None
        best_score = -1
        for match in matches:
            obj = torch.load(match, weights_only=False, map_location="cpu")
            value = obj["value"] if isinstance(obj, dict) and "value" in obj else obj
            score = _score_dump_candidate(name, value)
            if score > best_score or (
                score == best_score
                and best_path is not None
                and _extract_dump_index(match) > _extract_dump_index(best_path)
            ):
                best_score = score
                best_path = match

        if best_path is not None:
            resolved[name] = _extract_dump_index(best_path)
        elif name not in resolved:
            resolved[name] = -1
    return resolved


def load_value(d: Path, name: str, idx: int) -> torch.Tensor:
    if idx <= 0:
        raise FileNotFoundError(f"missing dump for {name} under {d}")
    p = d / f"forward_pass_id=0___rank=0___name={name}___dump_index={idx}.pt"
    obj = torch.load(p, weights_only=False, map_location="cpu")
    return obj["value"] if isinstance(obj, dict) and "value" in obj else obj


def align_single_step(name: str, x: torch.Tensor) -> torch.Tensor:
    if not ALIGN_TO_SINGLE_STEP or name not in ALIGN_NAMES:
        return x
    if x.ndim == 3:
        return x[:, -1, :]
    if x.ndim == 2 and x.shape[0] > 1:
        return x[-1:, :]
    return x


def normalize_for_compare(name: str, x: torch.Tensor, side: str) -> torch.Tensor:
    if name in SQUEEZE_BATCH1_NAMES and x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]
    if side == "hf" and name in HF_DROP_LAST_TOKEN_NAMES:
        if x.ndim == 1 and x.shape[0] > 1:
            x = x[:-1]
        elif x.ndim == 2 and x.shape[0] > 1:
            x = x[:-1, :]
    return x


def compare(
    logger: Logger,
    name: str,
    hf_dir: Path,
    sg_dir: Path,
    hf_index: dict[str, int],
    sg_index: dict[str, int],
) -> None:
    hf_idx = hf_index.get(name, -1)
    sg_idx = sg_index.get(name, -1)
    hf_name = HF_NAME_OVERRIDE.get(name, name)
    sg_name = SG_NAME_OVERRIDE.get(name, name)

    logger.log(f"\n[{name}]")
    if hf_idx <= 0 or sg_idx <= 0:
        logger.log(f"  -> skip (index not set): hf={hf_idx}, sg={sg_idx}")
        return

    try:
        x_hf = load_value(hf_dir, hf_name, hf_idx)
    except FileNotFoundError as e:
        logger.log(f"  -> HF file missing, skip: {e}")
        return
    try:
        x_sg = load_value(sg_dir, sg_name, sg_idx)
    except FileNotFoundError as e:
        logger.log(f"  -> SG file missing, skip: {e}")
        return

    x_hf = normalize_for_compare(name, x_hf, side="hf")
    x_sg = normalize_for_compare(name, x_sg, side="sg")
    x_hf = align_single_step(name, x_hf)
    x_sg = align_single_step(name, x_sg)

    logger.log(f"  hf shape/dtype: {tuple(x_hf.shape)} {x_hf.dtype}")
    logger.log(f"  sg shape/dtype: {tuple(x_sg.shape)} {x_sg.dtype}")
    logger.log(f"  shape_equal: {x_hf.shape == x_sg.shape}")
    logger.log(f"  dtype_equal: {x_hf.dtype == x_sg.dtype}")

    if x_hf.shape != x_sg.shape:
        logger.log("  -> shape mismatch, skip")
        return

    if x_hf.dtype == x_sg.dtype:
        logger.log(f"  torch.equal: {torch.equal(x_hf, x_sg)}")
    else:
        logger.log("  torch.equal: skip (dtype mismatch)")

    if x_hf.dtype.is_floating_point or x_sg.dtype.is_floating_point:
        x_hf_f = x_hf.float()
        x_sg_f = x_sg.float()
        diff = (x_hf_f - x_sg_f).abs()
        logger.log(f"  value_equal_after_cast: {bool(torch.equal(x_hf_f, x_sg_f))}")
        logger.log(f"  max_abs: {diff.max().item()}")
        logger.log(f"  mean_abs: {diff.mean().item()}")
    else:
        neq = (x_hf != x_sg).sum().item()
        logger.log(f"  value_equal_after_cast: {neq == 0}")
        logger.log(f"  neq_cnt: {neq}")


def main() -> None:
    args = parse_args()
    compare_names = FOCUS_TO_NAMES[args.focus]

    if args.hf_index_json is not None:
        hf_index = _load_index_map(args.hf_index_json, DEFAULT_HF_INDEX)
    elif args.auto_index:
        hf_index = _discover_index_map(args.hf_dir, compare_names, "hf", DEFAULT_HF_INDEX)
    else:
        hf_index = dict(DEFAULT_HF_INDEX)

    if args.sg_index_json is not None:
        sg_index = _load_index_map(args.sg_index_json, DEFAULT_SG_INDEX)
    elif args.auto_index:
        sg_index = _discover_index_map(args.sg_dir, compare_names, "sg", DEFAULT_SG_INDEX)
    else:
        sg_index = dict(DEFAULT_SG_INDEX)

    logger = Logger()
    logger.log(f"HF_DIR = {args.hf_dir}")
    logger.log(f"SG_DIR = {args.sg_dir}")
    logger.log(f"ALIGN_TO_SINGLE_STEP = {ALIGN_TO_SINGLE_STEP}")
    logger.log(f"FOCUS = {args.focus}")

    for name in compare_names:
        title = SECTION_TITLES.get(name)
        if title is not None:
            logger.log(f"\n== {title} ==")
        compare(logger, name, args.hf_dir, args.sg_dir, hf_index, sg_index)

    logger.flush_to(args.output_txt)


if __name__ == "__main__":
    main()
