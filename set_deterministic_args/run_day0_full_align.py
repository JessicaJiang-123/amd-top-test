#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
import torch


COMPARE_NAMES = [
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

HF_NAME_OVERRIDE = {
    "layer0_attn_after_input_layernorm_only": "layer0_attn_input_after_prepare",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click day0 deterministic alignment runner."
    )
    parser.add_argument("--server-gpu", type=str, default="4")
    parser.add_argument("--hf-gpu", type=str, default="5")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--server-attention-backend", type=str, default="triton")
    parser.add_argument("--hf-attn-implementation", type=str, default="triton")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--strict-bitwise", action="store_true")
    parser.add_argument("--use-batch-invariant", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("/app/results/day0_runs"))
    parser.add_argument("--server-start-timeout", type=int, default=180)
    parser.add_argument("--server-stop-timeout", type=int, default=20)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument(
        "--compare-script",
        type=Path,
        default=Path("/app/true_on_policy/experiment/step1_compare/compare_hf_sglang_step3_attn.py"),
    )
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pythonpath() -> str:
    base = "/app/sglang/python:/app/true_on_policy/miles"
    current = os.environ.get("PYTHONPATH", "")
    return f"{base}:{current}" if current else base


def make_env(cuda_visible_devices: str, dumper_root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["PYTHONPATH"] = build_pythonpath()
    env["SGLANG_DUMPER_ENABLE"] = "1"
    env["SGLANG_DUMPER_DIR"] = str(dumper_root)
    env["SGLANG_DUMPER_WRITE_FILE"] = "1"
    if extra:
        env.update(extra)
    return env


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def shell_join(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _prefix_writer(prefix: str, line: str) -> str:
    return f"{prefix}{line}" if prefix else line


def _stream_subprocess_output(
    proc: subprocess.Popen,
    log_path: Path,
    prefix: str = "",
) -> threading.Thread:
    def _reader() -> None:
        with log_path.open("w", encoding="utf-8") as log_f:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(_prefix_writer(prefix, line))
                sys.stdout.flush()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def run_cmd_with_tee(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    cwd: str = "/app",
    prefix: str = "",
) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
        text=True,
        bufsize=1,
    )
    reader = _stream_subprocess_output(proc, log_path, prefix=prefix)
    returncode = proc.wait()
    reader.join(timeout=1)
    return returncode


def wait_for_server(host: str, port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    base_url = f"http://{host}:{port}"
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code < 500:
                return
        except Exception as e:
            last_error = e
        try:
            with socket.create_connection((host, port), timeout=1):
                time.sleep(1)
                return
        except OSError as e:
            last_error = e
        time.sleep(1)
    raise RuntimeError(f"Server not ready within {timeout_s}s: {last_error}")


def stop_process(proc: subprocess.Popen, timeout_s: int) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=timeout_s)


def parse_partial_name(log_path: Path) -> str | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Choose partial_name=([0-9.]+)", text)
    return matches[0] if matches else None


def dir_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def resolve_dump_dirs(dumper_root: Path, server_log: Path, day0_log: Path) -> tuple[Path, Path]:
    sg_partial = parse_partial_name(server_log)
    hf_partial = parse_partial_name(day0_log)

    sg_dir = dumper_root / f"sglang_dump_{sg_partial}" if sg_partial else None
    hf_dir = dumper_root / f"sglang_dump_{hf_partial}" if hf_partial else None

    if sg_dir and hf_dir and sg_dir.exists() and hf_dir.exists():
        return hf_dir, sg_dir

    candidates = sorted(
        [p for p in dumper_root.glob("sglang_dump_*") if p.is_dir()],
        key=dir_mtime,
    )
    if len(candidates) < 2:
        raise RuntimeError(f"Expected at least 2 dump dirs under {dumper_root}, found {len(candidates)}")

    if sg_dir is None or not sg_dir.exists():
        sg_dir = candidates[0]
    if hf_dir is None or not hf_dir.exists():
        hf_dir = candidates[-1]
        if hf_dir == sg_dir and len(candidates) >= 2:
            hf_dir = candidates[-2]

    if hf_dir == sg_dir:
        raise RuntimeError(f"Failed to resolve distinct HF/SG dirs under {dumper_root}")
    return hf_dir, sg_dir


def extract_dump_index(path: Path) -> int:
    return int(path.stem.split("___dump_index=")[1])


def load_dump_value(path: Path):
    obj = torch.load(path, weights_only=False, map_location="cpu")
    return obj["value"] if isinstance(obj, dict) and "value" in obj else obj


def score_dump_candidate(name: str, value) -> int:
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


def resolve_index_map(dump_dir: Path, side: str) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for name in COMPARE_NAMES:
        dump_name = HF_NAME_OVERRIDE.get(name, name) if side == "hf" else name
        matches = sorted(
            dump_dir.glob(f"forward_pass_id=0___rank=0___name={dump_name}___dump_index=*.pt"),
            key=extract_dump_index,
        )
        if not matches:
            resolved[name] = -1
            continue

        best_path = None
        best_score = -1
        for match in matches:
            try:
                value = load_dump_value(match)
            except Exception:
                continue
            score = score_dump_candidate(name, value)
            if score > best_score or (
                score == best_score
                and best_path is not None
                and extract_dump_index(match) > extract_dump_index(best_path)
            ):
                best_score = score
                best_path = match

        resolved[name] = extract_dump_index(best_path or matches[0])
    return resolved


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--attention-backend",
        args.server_attention_backend,
        "--mem-fraction-static",
        "0.7",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--enable-deterministic-inference",
        "--disable-radix-cache",
        "--rl-on-policy-target",
        "fsdp",
        "--skip-server-warmup",
        "--disable-cuda-graph",
    ]


def build_day0_cmd(
    args: argparse.Namespace,
    day0_manifest_json: Path,
    day0_compare_json: Path,
    day0_compare_detail_txt: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "/app/true_on_policy/experiment/set_deterministic_args/day0_deterministic_align.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-path",
        args.model_path,
        "--attn-implementation",
        args.hf_attn_implementation,
        "--dtype",
        args.dtype,
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--manifest-json",
        str(day0_manifest_json),
        "--output-json",
        str(day0_compare_json),
        "--save-detail",
        str(day0_compare_detail_txt),
    ]
    if args.strict_bitwise:
        cmd.append("--strict-bitwise")
    if args.use_batch_invariant:
        cmd.append("--use-batch-invariant")
    return cmd


def build_compare_cmd(
    compare_script: Path,
    hf_dir: Path,
    sg_dir: Path,
    hf_index_json: Path,
    sg_index_json: Path,
) -> list[str]:
    return [
        sys.executable,
        str(compare_script),
        "--hf-dir",
        str(hf_dir),
        "--sg-dir",
        str(sg_dir),
        "--hf-index-json",
        str(hf_index_json),
        "--sg-index-json",
        str(sg_index_json),
        "--focus",
        "full",
    ]


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / f"run_{now_stamp()}"
    dumper_root = run_dir / "dumper"
    ensure_dir(dumper_root)

    server_log = run_dir / "server_stdout.txt"
    day0_log = run_dir / "day0_stdout.txt"
    compare_log = run_dir / "compare_stdout.txt"
    commands_txt = run_dir / "commands.txt"
    day0_manifest_json = run_dir / "manifest.json"
    day0_compare_json = run_dir / "compare.json"
    day0_compare_detail_txt = run_dir / "compare_detail.txt"
    hf_dir_txt = run_dir / "hf_dir.txt"
    sg_dir_txt = run_dir / "sg_dir.txt"
    hf_index_json = run_dir / "hf_index.json"
    sg_index_json = run_dir / "sg_index.json"
    manifest_json = run_dir / "run_manifest.json"

    server_cmd = build_server_cmd(args)
    day0_cmd = build_day0_cmd(
        args,
        day0_manifest_json,
        day0_compare_json,
        day0_compare_detail_txt,
    )

    write_text(
        commands_txt,
        "\n".join(
            [
                "[server]",
                shell_join(server_cmd),
                "",
                "[day0]",
                shell_join(day0_cmd),
                "",
                "[compare]",
                "<pending until hf/sg dirs and index json are resolved>",
                "",
            ]
        ),
    )

    server_env = make_env(
        args.server_gpu,
        dumper_root,
        {"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"},
    )
    day0_env = make_env(args.hf_gpu, dumper_root)

    server_proc = None
    server_reader = None
    compare_returncode = None
    day0_returncode = None
    server_returncode = None
    hf_dir = None
    sg_dir = None

    try:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd="/app",
            env=server_env,
            text=True,
            bufsize=1,
        )
        server_reader = _stream_subprocess_output(server_proc, server_log, prefix="[server] ")

        wait_for_server(args.host, args.port, args.server_start_timeout)

        day0_returncode = run_cmd_with_tee(
            day0_cmd,
            day0_env,
            day0_log,
            cwd="/app",
            prefix="[day0] ",
        )
        if day0_returncode not in (0, 1):
            raise RuntimeError(f"day0_deterministic_align failed with code {day0_returncode}")
        if day0_returncode == 1:
            print("[runner] day0 returned 1 because the compare found mismatches; continuing to dump compare artifacts.")

        hf_dir, sg_dir = resolve_dump_dirs(dumper_root, server_log, day0_log)
        write_text(hf_dir_txt, str(hf_dir) + "\n")
        write_text(sg_dir_txt, str(sg_dir) + "\n")

        hf_index = resolve_index_map(hf_dir, "hf")
        sg_index = resolve_index_map(sg_dir, "sg")
        write_json(hf_index_json, hf_index)
        write_json(sg_index_json, sg_index)

        compare_cmd = build_compare_cmd(args.compare_script, hf_dir, sg_dir, hf_index_json, sg_index_json)
        write_text(
            commands_txt,
            "\n".join(
                [
                    "[server]",
                    shell_join(server_cmd),
                    "",
                    "[day0]",
                    shell_join(day0_cmd),
                    "",
                    "[compare]",
                    shell_join(compare_cmd),
                    "",
                ]
            ),
        )
        compare_env = os.environ.copy()
        compare_env["PYTHONPATH"] = build_pythonpath()
        compare_returncode = run_cmd_with_tee(
            compare_cmd,
            compare_env,
            compare_log,
            cwd="/app",
            prefix="[compare] ",
        )
        if compare_returncode != 0:
            raise RuntimeError(f"compare script failed with code {compare_returncode}")
    finally:
        if server_proc is not None:
            if args.keep_server:
                server_returncode = server_proc.poll()
            else:
                stop_process(server_proc, args.server_stop_timeout)
                server_returncode = server_proc.poll()
            if server_reader is not None:
                try:
                    server_reader.join(timeout=1)
                except Exception:
                    pass

        manifest = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(run_dir),
            "dumper_root": str(dumper_root),
            "server_cmd": server_cmd,
            "day0_cmd": day0_cmd,
            "compare_cmd": build_compare_cmd(
                args.compare_script,
                hf_dir if hf_dir is not None else Path(""),
                sg_dir if sg_dir is not None else Path(""),
                hf_index_json,
                sg_index_json,
            ) if hf_dir is not None and sg_dir is not None else None,
            "server_log": str(server_log),
            "day0_log": str(day0_log),
            "compare_log": str(compare_log),
            "day0_manifest_json": str(day0_manifest_json),
            "day0_compare_json": str(day0_compare_json),
            "day0_compare_detail_txt": str(day0_compare_detail_txt),
            "hf_dir": str(hf_dir) if hf_dir is not None else None,
            "sg_dir": str(sg_dir) if sg_dir is not None else None,
            "hf_index_json": str(hf_index_json),
            "sg_index_json": str(sg_index_json),
            "server_returncode": server_returncode,
            "day0_returncode": day0_returncode,
            "compare_returncode": compare_returncode,
            "keep_server": args.keep_server,
        }
        write_json(manifest_json, manifest)

    print(f"run_dir: {run_dir}")
    print(f"server_log: {server_log}")
    print(f"day0_log: {day0_log}")
    print(f"compare_log: {compare_log}")
    print(f"commands_txt: {commands_txt}")
    print(f"day0_manifest_json: {day0_manifest_json}")
    print(f"day0_compare_json: {day0_compare_json}")
    print(f"day0_compare_detail_txt: {day0_compare_detail_txt}")
    print(f"hf_dir: {hf_dir}")
    print(f"sg_dir: {sg_dir}")
    print(f"hf_index_json: {hf_index_json}")
    print(f"sg_index_json: {sg_index_json}")


if __name__ == "__main__":
    main()
