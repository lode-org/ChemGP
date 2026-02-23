#!/usr/bin/env bash
# REAL-1 Part B: HCN 2D landscape via rgpycrumbs plt-neb
#
# Reads the cached HDF5 output from hcn_neb_profile.jl and generates
# a 2D RMSD landscape plot with structure insets at critical points.
#
# Requires: rgpycrumbs[surfaces,analysis], chemparseplot[plot,neb]
# Install: uv pip install -e /path/to/rgpycrumbs[surfaces,analysis] \
#                         -e /path/to/chemparseplot[plot,neb]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${CHEMGP_FIG_OUTPUT:-${SCRIPT_DIR}/output}"

# Default to standard NEB history; override with CHEMGP_H5_FILE
H5_FILE="${CHEMGP_H5_FILE:-${REPO_ROOT}/results_hcn_neb/standard/neb_history.h5}"

if [ ! -f "${H5_FILE}" ]; then
    echo "HDF5 file not found: ${H5_FILE}"
    echo "Run NEB with HDF5 output first (e.g. hcn_neb_profile.jl with PET-MAD server)."
    exit 1
fi

# rgpycrumbs and chemparseplot paths (override via env)
RGPYCRUMBS="${RGPYCRUMBS_PATH:-${HOME}/Git/Github/Python/rgpycrumbs}"
CHEMPARSEPLOT="${CHEMPARSEPLOT_PATH:-${HOME}/Git/Github/Python/chemparseplot}"

mkdir -p "${OUTPUT_DIR}"

cd "${RGPYCRUMBS}"
uv run --with "${CHEMPARSEPLOT}" rgpycrumbs/eon/plt_neb.py \
    --source hdf5 --input-h5 "${H5_FILE}" \
    --output-file "${OUTPUT_DIR}/hcn_landscape.pdf" \
    --plot-type landscape --rc-mode rmsd --show-pts \
    --landscape-path all --plot-structures crit_points \
    --surface-type grad_imq --theme ruhi \
    --facecolor white --figsize 5.37 5.37 --dpi 300 \
    --zoom-ratio 0.25 --fontsize-base 12 --title "" \
    --ase-rotation 0x,0y,0z --ira-kmax 14 --show-legend

echo "Saved: ${OUTPUT_DIR}/hcn_landscape.pdf"
