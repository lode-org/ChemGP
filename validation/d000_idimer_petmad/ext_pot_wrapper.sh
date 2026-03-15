#!/usr/bin/env bash
# Shell wrapper for eOn ext_pot: uses cached metatrain Python environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/ext_pot_petmad.py"
