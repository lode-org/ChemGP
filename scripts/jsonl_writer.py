#!/usr/bin/env python3
"""TCP JSONL writer for ChemGP optimizers.

Listens on a TCP socket for newline-terminated JSON lines from Julia
GP optimizers. Writes JSONL to a file and renders human-readable
summaries to stdout, both flushed immediately.

Protocol: each message is a single JSON object followed by newline.
The writer detects iteration lines vs summary lines by content.

Usage:
    python jsonl_writer.py --port 9876 --output convergence.jsonl

    # From Julia:
    #   sock = connect("localhost", 9876)
    #   println(sock, json_string)
    #   close(sock)  # or keep open for multiple lines
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading


def render_iter(data):
    """Render an iteration line as human-readable text."""
    gate = data.get("gate", "ok")
    gate_str = f" [{gate}]" if gate != "ok" else ""
    ls = data.get("ls", [])
    ls_str = ", ".join(f"{v:.2e}" for v in ls[:3])
    if len(ls) > 3:
        ls_str += ", ..."
    return (
        f"  iter {data['i']:3d} | E={data['E']:10.4f} | F={data['F']:.5f} | "
        f"oc={data['oc']:3d} | tp={data['tp']:3d} | t={data['t']:.2f}s | "
        f"sv={data['sv']:.2e} | ls=[{ls_str}] | td={data['td']:.4f}{gate_str}"
    )


def render_summary(data):
    """Render a summary line as human-readable text."""
    return (
        f"\n{'=' * 60}\n"
        f"  {data['status']} | oc={data['oc']} | "
        f"E={data['E']:.6f} | F={data['F']:.6f} | "
        f"iters={data['iters']}\n"
        f"{'=' * 60}"
    )


def handle_client(conn, addr, outfile, lock):
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
                    # Write JSONL
                    outfile.write(json_str + "\n")
                    outfile.flush()
                    os.fsync(outfile.fileno())

                    # Render human-readable
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


def main():
    parser = argparse.ArgumentParser(
        description="TCP JSONL writer for ChemGP optimizers"
    )
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--output", required=True, help="JSONL output file path")
    args = parser.parse_args()

    outfile = open(args.output, "w")
    lock = threading.Lock()

    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)  # dual-stack
    srv.bind(("::", args.port))
    srv.listen(4)
    srv.settimeout(1.0)

    print(f"JSONL writer on [::]:{args.port} -> {args.output}")
    sys.stdout.flush()

    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        while running:
            try:
                conn, addr = srv.accept()
                t = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, outfile, lock),
                    daemon=True,
                )
                t.start()
            except socket.timeout:
                continue
    finally:
        srv.close()
        outfile.close()


if __name__ == "__main__":
    main()
