#!/usr/bin/env python3
"""Figure pipeline orchestrator for ChemGP tutorial figures.

Manages the lifecycle of a generator (Julia, data production) and plotter
(Julia, PDF rendering) with real-time metrics via a TCP JSONL socket.

Architecture:
    Julia gen script
      |
      +-- TCP socket (metrics only) --> this orchestrator --> stdout + JSONL
      |
      +-- HDF5 file (all data) ------> Julia plot script ----> PDF

Usage:
    python figure_runner.py \
      --gen generators/gen_leps_minimize.jl \
      --plot plotters/plot_leps_minimize.jl \
      [--port 0]              # 0 = ephemeral (default)
      [--output-dir output/]
      [--julia-project .]
      [--skip-gen]            # reuse existing HDF5, jump to plot
      [--skip-plot]           # generate data only
      [--env KEY=VAL ...]     # extra env vars for subprocesses
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading


def derive_stem(gen_path):
    """Derive output stem from generator script name.

    gen_leps_minimize.jl -> leps_minimize
    """
    base = os.path.basename(gen_path)
    base = os.path.splitext(base)[0]
    return re.sub(r"^gen_", "", base)


def render_iter(data):
    """Render an iteration line as human-readable text."""
    gate = data.get("gate", "ok")
    gate_str = f" [{gate}]" if gate != "ok" else ""
    ls = data.get("ls", [])
    ls_str = ", ".join(f"{v:.2e}" for v in ls[:3])
    if len(ls) > 3:
        ls_str += ", ..."

    parts = []
    parts.append(f"iter {data.get('i', '?'):>3}")
    if "E" in data:
        parts.append(f"E={data['E']:10.4f}")
    if "F" in data:
        parts.append(f"F={data['F']:.5f}")
    if "max_force" in data:
        parts.append(f"F={data['max_force']:.5f}")
    if "oc" in data:
        parts.append(f"oc={data['oc']:>3}")
    if "oracle_calls" in data:
        parts.append(f"oc={data['oracle_calls']:>3}")
    if "tp" in data:
        parts.append(f"tp={data['tp']:>3}")
    if "t" in data:
        parts.append(f"t={data['t']:.2f}s")
    if "sv" in data:
        parts.append(f"sv={data['sv']:.2e}")
    if ls:
        parts.append(f"ls=[{ls_str}]")
    if "td" in data:
        parts.append(f"td={data['td']:.4f}")
    if "method" in data:
        parts.append(data["method"])

    return "  " + " | ".join(parts) + gate_str


def render_summary(data):
    """Render a summary line as human-readable text."""
    parts = [data.get("status", "DONE")]
    if "oc" in data:
        parts.append(f"oc={data['oc']}")
    if "E" in data:
        parts.append(f"E={data['E']:.6f}")
    if "F" in data:
        parts.append(f"F={data['F']:.6f}")
    if "iters" in data:
        parts.append(f"iters={data['iters']}")
    body = " | ".join(parts)
    return f"\n{'=' * 60}\n  {body}\n{'=' * 60}"


def handle_client(conn, outfile, lock):
    """Handle a single client connection (may send multiple lines)."""
    buf = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                json_str = json.dumps(data, separators=(",", ":"))

                with lock:
                    outfile.write(json_str + "\n")
                    outfile.flush()
                    os.fsync(outfile.fileno())

                    try:
                        if "status" in data:
                            print(render_summary(data))
                        elif "i" in data:
                            print(render_iter(data))
                    except (KeyError, ValueError):
                        pass
                    sys.stdout.flush()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        conn.close()


def run_socket_server(srv, outfile, lock, stop_event):
    """Accept connections until stop_event is set."""
    srv.settimeout(1.0)
    while not stop_event.is_set():
        try:
            conn, _addr = srv.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, outfile, lock),
                daemon=True,
            )
            t.start()
        except socket.timeout:
            continue


def build_env(base_env, port, output_dir, stem, extra_env):
    """Build subprocess environment with CHEMGP_FIG_* vars."""
    env = dict(base_env)
    env["CHEMGP_FIG_PORT"] = str(port)
    env["CHEMGP_FIG_OUTPUT"] = output_dir
    env["CHEMGP_FIG_STEM"] = stem
    env["CHEMGP_FIG_H5"] = os.path.join(output_dir, stem + ".h5")
    for kv in extra_env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    return env


def run_julia(script, project, env, label):
    """Run a Julia script as a subprocess, streaming its stdout/stderr."""
    cmd = ["julia", f"--project={project}", script]
    print(f"\n--- {label}: {' '.join(cmd)} ---")
    sys.stdout.flush()

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"*** {label} exited with code {proc.returncode}", file=sys.stderr)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Figure pipeline orchestrator for ChemGP tutorial figures"
    )
    parser.add_argument(
        "--gen",
        required=True,
        help="Path to generator Julia script (relative to tutorial dir)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Path to plotter Julia script (relative to tutorial dir)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port for metrics socket (0 = ephemeral, default)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <tutorial>/output/)",
    )
    parser.add_argument(
        "--julia-project",
        default=None,
        help="Julia project path (default: tutorial dir)",
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Skip generator, reuse existing HDF5",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotter, generate data only",
    )
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="Extra env vars as KEY=VAL pairs",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    tutorial_dir = os.path.dirname(os.path.abspath(__file__))
    gen_path = os.path.join(tutorial_dir, args.gen)
    plot_path = (
        os.path.join(tutorial_dir, args.plot) if args.plot else None
    )
    output_dir = args.output_dir or os.path.join(tutorial_dir, "output")
    julia_project = args.julia_project or tutorial_dir

    os.makedirs(output_dir, exist_ok=True)

    stem = derive_stem(args.gen)
    jsonl_path = os.path.join(output_dir, stem + ".jsonl")
    h5_path = os.path.join(output_dir, stem + ".h5")

    # --- Generator phase ---
    gen_rc = 0
    if not args.skip_gen:
        if not os.path.isfile(gen_path):
            print(f"Error: generator not found: {gen_path}", file=sys.stderr)
            sys.exit(1)

        # Create dual-stack TCP listener
        srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        srv.bind(("::", args.port))
        srv.listen(4)
        actual_port = srv.getsockname()[1]

        print(f"Metrics socket on [::]:{actual_port}")
        print(f"Output stem: {stem}")
        print(f"JSONL log: {jsonl_path}")
        print(f"HDF5 data: {h5_path}")
        sys.stdout.flush()

        outfile = open(jsonl_path, "w")
        lock = threading.Lock()
        stop_event = threading.Event()

        # Start socket server thread
        server_thread = threading.Thread(
            target=run_socket_server,
            args=(srv, outfile, lock, stop_event),
            daemon=True,
        )
        server_thread.start()

        # Build env and run generator
        env = build_env(os.environ, actual_port, output_dir, stem, args.env)
        gen_rc = run_julia(gen_path, julia_project, env, "Generator")

        # Shut down socket server
        stop_event.set()
        server_thread.join(timeout=5)
        srv.close()
        outfile.close()

        if gen_rc != 0:
            print(f"\nGenerator failed (exit {gen_rc}). Partial JSONL kept at {jsonl_path}")
            if args.plot and not args.skip_plot:
                print("Skipping plotter due to generator failure.")
            sys.exit(gen_rc)
    else:
        print(f"Skipping generator (--skip-gen). Expecting HDF5 at {h5_path}")
        sys.stdout.flush()

    # --- Plotter phase ---
    if plot_path and not args.skip_plot:
        if not os.path.isfile(plot_path):
            print(f"Error: plotter not found: {plot_path}", file=sys.stderr)
            sys.exit(1)

        env = build_env(os.environ, 0, output_dir, stem, args.env)
        plot_rc = run_julia(plot_path, julia_project, env, "Plotter")

        if plot_rc != 0:
            print(f"\nPlotter failed (exit {plot_rc}).")
            sys.exit(plot_rc)

    print(f"\nDone: {stem}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
