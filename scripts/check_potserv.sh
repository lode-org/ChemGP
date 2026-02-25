#!/usr/bin/env bash
# Check whether potserv (rgpot RPC server) is built and available.
# Prints what needs to be done, or confirms everything is ready.
#
# Usage: bash scripts/check_potserv.sh
#        pixi run bash scripts/check_potserv.sh   # inside pixi env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RGPOT_DIR="${REPO_ROOT}/rgpot"
BUILD_DIR="${RGPOT_DIR}/builddir"
POTSERV="${BUILD_DIR}/CppCore/rgpot/rpc/potserv"
BRIDGE="${BUILD_DIR}/CppCore/rgpot/rpc/libpot_client_bridge.so"

status=0

# 1. Check rgpot clone
if [ ! -d "${RGPOT_DIR}" ]; then
    echo "[MISSING] rgpot not cloned. Run:"
    echo "    pixi run rgpot-clone"
    status=1
else
    echo "[OK] rgpot cloned at ${RGPOT_DIR}"
fi

# 2. Check meson build directory
if [ ! -d "${BUILD_DIR}" ]; then
    echo "[MISSING] Build directory not configured. Run:"
    echo "    pixi run rgpot-setup"
    status=1
else
    echo "[OK] Build directory exists"
fi

# 3. Check potserv binary
if [ ! -x "${POTSERV}" ]; then
    echo "[MISSING] potserv binary not found. Run:"
    echo "    pixi run rgpot-build"
    status=1
else
    echo "[OK] potserv: ${POTSERV}"
fi

# 4. Check bridge library
if [ ! -f "${BRIDGE}" ]; then
    echo "[MISSING] libpot_client_bridge.so not found. Run:"
    echo "    pixi run rgpot-build"
    status=1
else
    echo "[OK] bridge: ${BRIDGE}"
fi

# 5. Check if potserv is currently running
if pgrep -x potserv >/dev/null 2>&1; then
    port=$(ss -tlnp 2>/dev/null | grep potserv | grep -oP ':\K[0-9]+' | head -1)
    echo "[RUNNING] potserv is active${port:+ on port ${port}}"
else
    echo "[STOPPED] potserv is not running. Start with:"
    echo "    ${POTSERV} <port> <potential>"
    echo "    e.g.: ${POTSERV} 12345 pet-mad"
fi

# Summary
echo ""
if [ "${status}" -eq 0 ]; then
    echo "All rgpot components are built and ready."
else
    echo "Some components are missing. Quick setup:"
    echo "    pixi run rgpot-build   # clones, configures, and builds in order"
fi

exit "${status}"
