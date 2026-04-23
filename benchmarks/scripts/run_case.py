#!/usr/bin/env python3
"""Run one benchmark case and capture structured metadata."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Benchmark YAML config")
    parser.add_argument("--task", required=True, help="Task family")
    parser.add_argument("--case", required=True, help="Case id")
    parser.add_argument("--variant", required=True, help="Benchmark variant")
    parser.add_argument("--output-jsonl", required=True, help="Raw JSONL output path")
    parser.add_argument("--output-meta", required=True, help="Run metadata JSON path")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_case(cfg: dict, task: str, case_id: str) -> dict:
    for case in cfg["tasks"][task]["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(f"Unknown benchmark case '{case_id}' for task '{task}'")


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


def wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if port_open(host, port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def start_rpc_server(
    repo_root: Path,
    log_path: Path,
    host: str,
    port: int,
    env: dict[str, str],
) -> subprocess.Popen[str] | None:
    if port_open(host, port):
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "pixi",
            "run",
            "-e",
            "rpc",
            "eonclient",
            "-p",
            "metatomic",
            "--config",
            "config/eon_serve_petmad.ini",
            "--serve-port",
            str(port),
        ],
        cwd=repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_port(host, port, timeout_s=180.0)
    except Exception:
        proc.terminate()
        proc.wait(timeout=30)
        raise
    return proc


def build_command(case: dict, task: str) -> list[str]:
    cmd = ["cargo", "run"]
    if task != "smoke":
        cmd.append("--release")
    features = case.get("features", [])
    if features:
        cmd.extend(["--features", ",".join(features)])
    cmd.extend(["--example", case["example"]])
    case_args = case.get("args", [])
    if case_args:
        cmd.append("--")
        cmd.extend(case_args)
    return cmd


def configure_local_rgpot(repo_root: Path, env: dict[str, str]) -> None:
    build_dir = Path(
        env.get(
            "RGPOT_BUILD_DIR",
            str(Path.home() / "Git/Github/OmniPotentRPC/rgpot-direct-metatomic-export/bbdir"),
        )
    ).resolve()
    if not build_dir.exists():
        raise FileNotFoundError(f"RGPOT_BUILD_DIR does not exist: {build_dir}")

    rgpot_root = build_dir.parent
    lib_dirs = [
        build_dir / "CppCore/rgpot/MetatomicPot",
        build_dir / "cargo-target" / "release",
        build_dir / "cargo-target" / "debug",
    ]
    patterns = [
        ".pixi/envs/metatomicbld/lib/python*/site-packages/torch/lib",
        ".pixi/envs/metatomicbld/lib/python*/site-packages/metatensor/lib",
        ".pixi/envs/metatomicbld/lib/python*/site-packages/metatensor/torch/torch-*/lib",
        ".pixi/envs/metatomicbld/lib/python*/site-packages/metatomic/torch/torch-*/lib",
        ".pixi/envs/metatomicbld/lib/python*/site-packages/vesin/lib",
    ]
    for pattern in patterns:
        lib_dirs.extend(sorted(rgpot_root.glob(pattern)))

    env["RGPOT_BUILD_DIR"] = str(build_dir)
    env.setdefault("RGPOT_MODEL_PATH", str((repo_root / "models" / "pet-mad-xs-v1.5.0.pt").resolve()))
    rustflags = env.get("RUSTFLAGS", "").split()
    if "-Clink-arg=-fuse-ld=bfd" not in rustflags:
        rustflags.append("-Clink-arg=-fuse-ld=bfd")
    env["RUSTFLAGS"] = " ".join(flag for flag in rustflags if flag)

    ld_dirs = [str(path) for path in lib_dirs if path.exists()]
    if ld_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(ld_dirs + ([existing] if existing else []))


def configure_release_linking(env: dict[str, str], task: str) -> None:
    if task == "smoke":
        return
    # Pixi's forced conda target linker breaks release executable links on rg.cosmolab.
    env.pop("CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER", None)
    env.setdefault("CC", "/usr/bin/cc")
    env.setdefault("CXX", "/usr/bin/c++")


def configure_tool_paths(env: dict[str, str]) -> None:
    home = Path.home()
    extra = [home / ".cargo" / "bin", home / ".pixi" / "bin"]
    existing = env.get("PATH", "")
    prefix = ":".join(str(path) for path in extra if path.exists())
    if prefix:
        env["PATH"] = prefix + (f":{existing}" if existing else "")


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    case = find_case(cfg, args.task, args.case)

    repo_root = Path(__file__).resolve().parents[2]
    output_jsonl = Path(args.output_jsonl).resolve()
    output_meta = Path(args.output_meta).resolve()
    run_log = output_meta.with_name("run.log")
    server_log = output_meta.with_name("rpc_server.log")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    configure_tool_paths(env)
    env["CHEMGP_BENCH_VARIANT"] = args.variant
    env["CHEMGP_BENCH_OUTPUT"] = str(output_jsonl)
    env["CHEMGP_BENCH_ARTIFACT_DIR"] = str(output_jsonl.parent)
    env.setdefault("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
    env.setdefault("RGPOT_HOST", "127.0.0.1")
    env.setdefault("RGPOT_PORT", "12345")
    env.setdefault(
        "CARGO_TARGET_DIR",
        str((repo_root / "target" / "benchmarks" / args.task / args.case).resolve()),
    )
    configure_release_linking(env, args.task)
    if case.get("requires_local_rgpot", False):
        configure_local_rgpot(repo_root, env)

    rpc_proc: subprocess.Popen[str] | None = None
    started_rpc_server = False
    command = build_command(case, args.task)
    start_time = time.time()
    exit_code = -1

    try:
        if case.get("requires_rpc", False):
            rpc_proc = start_rpc_server(
                repo_root,
                server_log,
                env["RGPOT_HOST"],
                int(env["RGPOT_PORT"]),
                env,
            )
            started_rpc_server = rpc_proc is not None

        with run_log.open("w", encoding="utf-8") as log_handle:
            log_handle.write("COMMAND: " + " ".join(command) + "\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        exit_code = completed.returncode
    finally:
        if rpc_proc is not None:
            rpc_proc.terminate()
            try:
                rpc_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rpc_proc.kill()
                rpc_proc.wait(timeout=30)

    runtime_seconds = time.time() - start_time

    meta = {
        "task": args.task,
        "case": args.case,
        "group": case["group"],
        "variant": args.variant,
        "example": case["example"],
        "runner_env": case.get("runner_env", "dev"),
        "features": case.get("features", []),
        "args": case.get("args", []),
        "requires_rpc": bool(case.get("requires_rpc", False)),
        "requires_local_rgpot": bool(case.get("requires_local_rgpot", False)),
        "started_rpc_server": started_rpc_server,
        "runtime_seconds": runtime_seconds,
        "exit_code": exit_code,
        "output_jsonl": str(output_jsonl),
        "run_log": str(run_log),
        "server_log": str(server_log) if case.get("requires_rpc", False) else None,
        "command": command,
    }

    with output_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")

    if exit_code != 0:
        return exit_code
    if not output_jsonl.exists():
        print(f"Missing benchmark output: {output_jsonl}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
