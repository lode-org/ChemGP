#!/usr/bin/env bash
# Shell wrapper for eOn ext_pot: uses cached metatrain Python environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /home/rgoswami/.cache/uv/archive-v0/8Pu-CjOc6Ob0c0mvgKD6u/bin/python "$SCRIPT_DIR/ext_pot_petmad.py"
