#!/usr/bin/env python3
"""Convert Rust example JSONL output to HDF5 for Julia plotters.

Reads *.jsonl in the project root (produced by cargo run --release --example ...),
writes *.h5 into scripts/figures/tutorial/output/ matching Julia plotter schemas.

Usage:
    uv run scripts/jsonl_to_h5.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "scripts" / "figures" / "tutorial" / "output"

# Display-name mapping for convergence plots
METHOD_LABELS = {
    "gp_minimize": "GP-minimization",
    "direct_minimize": "Classical L-BFGS",
    "standard_dimer": "Classical Dimer",
    "gp_dimer": "GP-Dimer",
    "otgpd": "OTGPD",
    "neb": "Standard NEB",
    "gp_neb_aie": "GP-NEB AIE",
    "gp_neb_oie": "GP-NEB OIE",
}


def label(method):
    return METHOD_LABELS.get(method, method)


# -- helpers for HDF5 writing --


def h5_write_table(f, name, columns):
    """Write a group of same-length vectors (table convention)."""
    g = f.require_group(name)
    for k, v in columns.items():
        arr = np.array(v)
        if k in g:
            del g[k]
        if arr.dtype.kind == "U":
            dt = h5py.string_dtype()
            g.create_dataset(k, data=arr.astype(object), dtype=dt)
        else:
            g.create_dataset(k, data=arr)


def h5_write_grid(f, name, data, x_range, y_range):
    """Write a 2D grid under /grids/<name> with range attributes."""
    grids = f.require_group("grids")
    if name in grids:
        del grids[name]
    ds = grids.create_dataset(name, data=data)
    ds.attrs["x_range"] = np.array(x_range, dtype=np.float64)
    ds.attrs["x_length"] = data.shape[1]
    ds.attrs["y_range"] = np.array(y_range, dtype=np.float64)
    ds.attrs["y_length"] = data.shape[0]


def h5_write_points(f, name, columns):
    """Write a point set under /points/<name>."""
    g = f.require_group("points").require_group(name)
    for k, v in columns.items():
        if k in g:
            del g[k]
        g.create_dataset(k, data=np.array(v, dtype=np.float64))


def h5_write_path(f, name, columns):
    """Write a path under /paths/<name>."""
    g = f.require_group("paths").require_group(name)
    for k, v in columns.items():
        if k in g:
            del g[k]
        g.create_dataset(k, data=np.array(v, dtype=np.float64))


# -- individual converters --


def convert_leps_minimize():
    """leps_minimize_comparison.jsonl -> leps_minimize.h5"""
    src = ROOT / "leps_minimize_comparison.jsonl"
    if not src.exists():
        return
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    oc, fatom, methods = [], [], []
    for r in records:
        oc.append(r["oracle_calls"])
        fatom.append(r.get("max_fatom", 0.0))
        methods.append(label(r["method"]))

    dst = OUTDIR / "leps_minimize.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": oc,
            "max_fatom": fatom,
            "method": methods,
        })
    print(f"Wrote {dst}")


def convert_leps_dimer():
    """leps_dimer_comparison.jsonl -> mb_dimer.h5 (plotter expects mb_dimer)."""
    src = ROOT / "leps_dimer_comparison.jsonl"
    if not src.exists():
        return
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    oc, fnorm, methods = [], [], []
    for r in records:
        if r.get("summary"):
            continue
        oc.append(r["oracle_calls"])
        fnorm.append(r["force"])
        methods.append(label(r["method"]))

    dst = OUTDIR / "mb_dimer.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": oc,
            "force_norm": fnorm,
            "method": methods,
        })
    print(f"Wrote {dst}")


def convert_leps_rff():
    """leps_rff_quality.jsonl -> leps_rff.h5"""
    src = ROOT / "leps_rff_quality.jsonl"
    if not src.exists():
        return
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    gp_e_mae = gp_g_mae = 0.0
    d_rff, e_true, g_true, e_gp, g_gp = [], [], [], [], []
    for r in records:
        if r.get("type") == "exact_gp":
            gp_e_mae = r["energy_mae"]
            gp_g_mae = r["gradient_mae"]
        elif r.get("type") == "rff":
            d_rff.append(r["d_rff"])
            e_true.append(r["energy_mae_vs_true"])
            g_true.append(r["gradient_mae_vs_true"])
            e_gp.append(r["energy_mae_vs_gp"])
            g_gp.append(r["gradient_mae_vs_gp"])

    dst = OUTDIR / "leps_rff.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "D_rff": d_rff,
            "energy_mae_vs_true": e_true,
            "gradient_mae_vs_true": g_true,
            "energy_mae_vs_gp": e_gp,
            "gradient_mae_vs_gp": g_gp,
        })
        f.attrs["gp_e_mae"] = gp_e_mae
        f.attrs["gp_g_mae"] = gp_g_mae
    print(f"Wrote {dst}")


def convert_leps_neb():
    """leps_neb_comparison.jsonl -> leps_aie_oie.h5 + leps_neb.h5"""
    src = ROOT / "leps_neb_comparison.jsonl"
    if not src.exists():
        return
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    # Convergence history for leps_aie_oie.h5
    oc, mf, methods = [], [], []
    # Grid data for leps_neb.h5
    grid_meta = None
    grid_data = []
    neb_path = []
    endpoints = []

    for r in records:
        if r.get("summary"):
            continue
        t = r.get("type")
        if t == "grid_meta":
            grid_meta = r
        elif t == "grid":
            grid_data.append(r)
        elif t == "neb_path":
            neb_path.append(r)
        elif t == "endpoint":
            endpoints.append(r)
        elif "method" in r:
            oc.append(r["oracle_calls"])
            mf.append(r["max_force"])
            methods.append(label(r["method"]))

    # leps_aie_oie.h5
    dst1 = OUTDIR / "leps_aie_oie.h5"
    with h5py.File(dst1, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": oc,
            "max_force": mf,
            "method": methods,
        })
    print(f"Wrote {dst1}")

    # leps_neb.h5 (energy grid + NEB path + endpoints)
    if grid_meta and grid_data:
        nx, ny = grid_meta["nx"], grid_meta["ny"]
        rab_range = (grid_meta["rab_min"], grid_meta["rab_max"])
        rbc_range = (grid_meta["rbc_min"], grid_meta["rbc_max"])

        energy = np.zeros((nx, ny))
        for r in grid_data:
            energy[r["ix"], r["iy"]] = r["energy"]

        dst2 = OUTDIR / "leps_neb.h5"
        with h5py.File(dst2, "w") as f:
            h5_write_grid(f, "energy", energy, rab_range, rbc_range)

            if neb_path:
                neb_path.sort(key=lambda r: r["image"])
                h5_write_path(f, "neb", {
                    "rAB": [r["rAB"] for r in neb_path],
                    "rBC": [r["rBC"] for r in neb_path],
                })

            if endpoints:
                h5_write_points(f, "endpoints", {
                    "rAB": [r["rAB"] for r in endpoints],
                    "rBC": [r["rBC"] for r in endpoints],
                })

        print(f"Wrote {dst2}")


def convert_mb_gp_quality():
    """mb_gp_quality.jsonl -> mb_gp.h5 + mb_variance.h5"""
    src = ROOT / "mb_gp_quality.jsonl"
    if not src.exists():
        return
    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    meta = None
    minima = []
    saddles = []
    train_points = defaultdict(list)
    grids = defaultdict(list)

    for r in records:
        t = r["type"]
        if t == "grid_meta":
            meta = r
        elif t == "minimum":
            minima.append(r)
        elif t == "saddle":
            saddles.append(r)
        elif t == "train_point":
            train_points[r["n_train"]].append(r)
        elif t == "grid":
            grids[r["n_train"]].append(r)

    if meta is None:
        return

    nx, ny = meta["nx"], meta["ny"]
    x_range = (meta["x_min"], meta["x_max"])
    y_range = (meta["y_min"], meta["y_max"])

    def grid_to_array(recs, field):
        arr = np.zeros((nx, ny))
        for r in recs:
            arr[r["ix"], r["iy"]] = r[field]
        return arr

    # mb_gp.h5: true_energy + gp_mean_N{} + train_N{}
    dst1 = OUTDIR / "mb_gp.h5"
    n_trains = sorted(grids.keys())
    with h5py.File(dst1, "w") as f:
        # True energy from any N (same for all)
        first_n = n_trains[0]
        true_e = grid_to_array(grids[first_n], "true_e")
        h5_write_grid(f, "true_energy", true_e, x_range, y_range)

        for n in n_trains:
            gp_e = grid_to_array(grids[n], "gp_e")
            h5_write_grid(f, f"gp_mean_N{n}", gp_e, x_range, y_range)

            pts = train_points[n]
            h5_write_points(f, f"train_N{n}", {
                "x": [p["x"] for p in pts],
                "y": [p["y"] for p in pts],
            })

    print(f"Wrote {dst1}")

    # mb_variance.h5: energy + variance at N=15 + points + metadata
    var_n = 15
    if var_n in grids:
        true_e = grid_to_array(grids[var_n], "true_e")
        gp_var = grid_to_array(grids[var_n], "gp_var")
        var_clip = float(np.percentile(gp_var, 95))
        max_idx = np.unravel_index(np.argmax(gp_var), gp_var.shape)
        xs = np.linspace(x_range[0], x_range[1], nx)
        ys = np.linspace(y_range[0], y_range[1], ny)
        max_var_x = float(xs[max_idx[0]])
        max_var_y = float(ys[max_idx[1]])

        dst2 = OUTDIR / "mb_variance.h5"
        with h5py.File(dst2, "w") as f:
            h5_write_grid(f, "energy", true_e, x_range, y_range)
            h5_write_grid(f, "variance", gp_var, x_range, y_range)

            pts = train_points[var_n]
            h5_write_points(f, "training", {
                "x": [p["x"] for p in pts],
                "y": [p["y"] for p in pts],
            })
            h5_write_points(f, "minima", {
                "x": [m["x"] for m in minima],
                "y": [m["y"] for m in minima],
            })
            h5_write_points(f, "saddles", {
                "x": [s["x"] for s in saddles],
                "y": [s["y"] for s in saddles],
            })

            f.attrs["max_var_x"] = max_var_x
            f.attrs["max_var_y"] = max_var_y
            f.attrs["var_clip"] = var_clip

        print(f"Wrote {dst2}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    convert_leps_minimize()
    convert_leps_dimer()
    convert_leps_rff()
    convert_leps_neb()
    convert_mb_gp_quality()


if __name__ == "__main__":
    main()
