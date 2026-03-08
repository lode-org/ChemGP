#!/usr/bin/env python3
"""Convert Rust example JSONL output to HDF5 for Julia plotters.

Each Rust example writes flat JSONL records. The Julia plotters in
scripts/figures/tutorial/plotters/ consume HDF5 with a specific schema
(grids/, paths/, points/ groups, root attributes). This script bridges
the two.

Usage:
    pixi run -e dev python scripts/jsonl_to_h5.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "scripts" / "figures" / "tutorial" / "output"

# Display-name mapping (must match Julia plotter filter strings)
METHOD_LABELS = {
    "gp_minimize": "GP-minimization",
    "direct_minimize": "Classical L-BFGS",
    "direct_lbfgs": "Classical L-BFGS",
    "direct_gd": "Classical L-BFGS",
    "standard_dimer": "Standard Dimer",
    "gp_dimer": "GP-Dimer",
    "otgpd": "OTGPD",
    "neb": "Standard NEB",
    "gp_neb_aie": "GP-NEB AIE",
    "gp_neb_oie": "GP-NEB OIE",
    "gp_neb_oie_naive": "GP-NEB OIE (naive)",
}


def label(method):
    return METHOD_LABELS.get(method, method)


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# -- HDF5 write helpers (match Julia common_data.jl schema) --


def h5_write_table(f, name, columns):
    """Write a group of same-length vectors (table convention)."""
    if name in f:
        del f[name]
    g = f.create_group(name)
    for k, v in columns.items():
        arr = np.array(v)
        if arr.dtype.kind == "U":
            dt = h5py.string_dtype()
            g.create_dataset(k, data=arr.astype(object), dtype=dt)
        else:
            g.create_dataset(k, data=arr)


def h5_write_grid(f, name, data, x_range, y_range):
    """Write 2D grid under /grids/<name> with range attributes.

    Data is transposed before writing: Julia HDF5.jl reverses dimensions
    when reading (Fortran vs C order), so writing data.T from Python
    ensures Julia gets data[ix, iy] with correct orientation.
    """
    grids = f.require_group("grids")
    if name in grids:
        del grids[name]
    ds = grids.create_dataset(name, data=data.T)
    ds.attrs["x_range"] = np.array(x_range, dtype=np.float64)
    ds.attrs["x_length"] = data.shape[0]
    ds.attrs["y_range"] = np.array(y_range, dtype=np.float64)
    ds.attrs["y_length"] = data.shape[1]


def h5_write_points(f, name, columns):
    """Write point sets under /points/<name>."""
    pts = f.require_group("points")
    if name in pts:
        del pts[name]
    g = pts.create_group(name)
    for k, v in columns.items():
        g.create_dataset(k, data=np.array(v, dtype=np.float64))


def h5_write_path(f, name, columns):
    """Write ordered path under /paths/<name>."""
    paths = f.require_group("paths")
    if name in paths:
        del paths[name]
    g = paths.create_group(name)
    for k, v in columns.items():
        g.create_dataset(k, data=np.array(v, dtype=np.float64))


def grid_to_array(recs, nx, ny, field):
    """Reconstruct 2D array from flat (ix, iy, value) JSONL records."""
    arr = np.zeros((nx, ny))
    for r in recs:
        arr[r["ix"], r["iy"]] = r[field]
    return arr


# -------------------------------------------------------------------
# Conversion functions
# -------------------------------------------------------------------


def convert_mb_gp_quality():
    """mb_gp_quality.jsonl -> mb_gp.h5 + mb_variance.h5"""
    src = ROOT / "mb_gp_quality.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    meta = None
    minima, saddles = [], []
    train_by_n = defaultdict(list)
    grid_by_n = defaultdict(list)

    for r in records:
        t = r.get("type")
        if t == "grid_meta":
            meta = r
        elif t == "minimum":
            minima.append(r)
        elif t == "saddle":
            saddles.append(r)
        elif t == "train_point":
            train_by_n[r["n_train"]].append(r)
        elif t == "grid":
            grid_by_n[r["n_train"]].append(r)

    if meta is None:
        return

    nx, ny = meta["nx"], meta["ny"]
    x_range = (meta["x_min"], meta["x_max"])
    y_range = (meta["y_min"], meta["y_max"])
    n_trains = sorted(grid_by_n.keys())

    # -- mb_gp.h5: true_energy + gp_mean_N{n} + train_N{n} --
    dst1 = OUTDIR / "mb_gp.h5"
    with h5py.File(dst1, "w") as f:
        true_e = grid_to_array(grid_by_n[n_trains[0]], nx, ny, "true_e")
        h5_write_grid(f, "true_energy", true_e, x_range, y_range)

        for n in n_trains:
            gp_e = grid_to_array(grid_by_n[n], nx, ny, "gp_e")
            h5_write_grid(f, f"gp_mean_N{n}", gp_e, x_range, y_range)

            pts = train_by_n[n]
            h5_write_points(f, f"train_N{n}", {
                "x": [p["x"] for p in pts],
                "y": [p["y"] for p in pts],
            })
    print(f"  wrote {dst1}")

    # -- mb_variance.h5: energy + variance at N=15 + stationary points --
    var_n = 15
    if var_n not in grid_by_n:
        var_n = n_trains[-1]

    true_e = grid_to_array(grid_by_n[var_n], nx, ny, "true_e")
    gp_var = grid_to_array(grid_by_n[var_n], nx, ny, "gp_var")

    var_clip = float(np.percentile(gp_var[gp_var > 0], 95)) if np.any(gp_var > 0) else 1.0
    max_idx = np.unravel_index(np.argmax(gp_var), gp_var.shape)
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    max_var_x = float(xs[max_idx[0]])
    max_var_y = float(ys[max_idx[1]])

    dst2 = OUTDIR / "mb_variance.h5"
    with h5py.File(dst2, "w") as f:
        h5_write_grid(f, "energy", true_e, x_range, y_range)
        h5_write_grid(f, "variance", gp_var, x_range, y_range)

        pts = train_by_n[var_n]
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
    print(f"  wrote {dst2}")


def convert_mb_gp_scattered():
    """mb_gp_scattered.jsonl -> mb_gp_scattered.h5

    Same schema as mb_gp.h5 (true_energy + gp_mean_N{n} + train_N{n}).
    """
    src = ROOT / "mb_gp_scattered.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    meta = None
    train_by_n = defaultdict(list)
    grid_by_n = defaultdict(list)

    for r in records:
        t = r.get("type")
        if t == "grid_meta":
            meta = r
        elif t == "train_point":
            train_by_n[r["n_train"]].append(r)
        elif t == "grid":
            grid_by_n[r["n_train"]].append(r)

    if meta is None:
        return

    nx, ny = meta["nx"], meta["ny"]
    x_range = (meta["x_min"], meta["x_max"])
    y_range = (meta["y_min"], meta["y_max"])
    n_trains = sorted(grid_by_n.keys())

    dst = OUTDIR / "mb_gp_scattered.h5"
    with h5py.File(dst, "w") as f:
        true_e = grid_to_array(grid_by_n[n_trains[0]], nx, ny, "true_e")
        h5_write_grid(f, "true_energy", true_e, x_range, y_range)

        for n in n_trains:
            gp_e = grid_to_array(grid_by_n[n], nx, ny, "gp_e")
            h5_write_grid(f, f"gp_mean_N{n}", gp_e, x_range, y_range)

            pts = train_by_n[n]
            h5_write_points(f, f"train_N{n}", {
                "x": [p["x"] for p in pts],
                "y": [p["y"] for p in pts],
            })
    print(f"  wrote {dst}")


def convert_mb_neb():
    """mb_neb.jsonl -> mb_neb.h5

    grids/energy, paths/neb {x,y}, points/minima {x,y}
    """
    src = ROOT / "mb_neb.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    meta = next(r for r in records if r.get("type") == "grid_meta")
    nx, ny = meta["nx"], meta["ny"]
    x_range = (meta["x_min"], meta["x_max"])
    y_range = (meta["y_min"], meta["y_max"])

    grid_recs = [r for r in records if r.get("type") == "grid"]
    energy = grid_to_array(grid_recs, nx, ny, "energy")

    neb_recs = sorted(
        [r for r in records if r.get("type") == "neb_path"],
        key=lambda r: r["image"],
    )
    minima = [r for r in records if r.get("type") == "minimum"]
    saddles = [r for r in records if r.get("type") == "saddle"]

    dst = OUTDIR / "mb_neb.h5"
    with h5py.File(dst, "w") as f:
        h5_write_grid(f, "energy", energy, x_range, y_range)
        h5_write_path(f, "neb", {
            "x": [r["x"] for r in neb_recs],
            "y": [r["y"] for r in neb_recs],
        })
        h5_write_points(f, "minima", {
            "x": [m["x"] for m in minima],
            "y": [m["y"] for m in minima],
        })
        h5_write_points(f, "saddles", {
            "x": [s["x"] for s in saddles],
            "y": [s["y"] for s in saddles],
        })
    print(f"  wrote {dst}")


def convert_mb_dimer():
    """mb_dimer_comparison.jsonl -> mb_dimer.h5

    table {oracle_calls, force_norm, method}
    """
    src = ROOT / "mb_dimer_comparison.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    all_records = read_jsonl(src)
    records = [r for r in all_records if "method" in r and not r.get("summary")]
    summary = next((r for r in all_records if r.get("summary")), {})

    dst = OUTDIR / "mb_dimer.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": [r["oracle_calls"] for r in records],
            "force_norm": [r["force"] for r in records],
            "method": [label(r["method"]) for r in records],
        })
        if "conv_tol" in summary:
            f.attrs["conv_tol"] = summary["conv_tol"]
    print(f"  wrote {dst}")


def convert_mb_hyperparams():
    """mb_hyperparams.jsonl -> mb_hyperparams.h5

    table/slice {x}, table/true_surface {E_true},
    table/gp_ls{j}_sv{i} {E_pred, E_std}
    """
    src = ROOT / "mb_hyperparams.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)
    preds = [r for r in records if r.get("type") == "prediction"]

    x_vals = sorted(set(r["x"] for r in preds))
    ells = sorted(set(r["ell"] for r in preds))
    sf2s = sorted(set(r["sigma_f2"] for r in preds))

    # Group predictions by (sigma_f2, ell) -> {x: record}
    by_combo = defaultdict(dict)
    for r in preds:
        by_combo[(r["sigma_f2"], r["ell"])][r["x"]] = r

    dst = OUTDIR / "mb_hyperparams.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "slice", {"x": x_vals})

        # True surface (same for all combos)
        first = next(iter(by_combo.values()))
        h5_write_table(f, "true_surface", {
            "E_true": [first[x]["true_e"] for x in x_vals],
        })

        # 3x3 grid: j indexes ell (columns), i indexes sigma_f2 (rows)
        for i_idx, sf2 in enumerate(sf2s, 1):
            for j_idx, ell in enumerate(ells, 1):
                combo = by_combo[(sf2, ell)]
                name = f"gp_ls{j_idx}_sv{i_idx}"
                h5_write_table(f, name, {
                    "E_pred": [combo[x]["gp_mean"] for x in x_vals],
                    "E_std": [np.sqrt(max(combo[x]["gp_var"], 0.0)) for x in x_vals],
                })
    print(f"  wrote {dst}")


def convert_mb_trust():
    """mb_trust.jsonl -> mb_trust.h5

    table/slice {x, E_true, E_pred, E_std, dist_to_data, in_trust},
    points/training {x, y}, attrs: trust_radius, y_slice
    """
    src = ROOT / "mb_trust.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    trust_meta = next(r for r in records if r.get("type") == "trust_meta")
    trust_r = trust_meta["trust_radius"]
    y_slice = 0.5  # known from mb_trust.rs

    train_pts = [r for r in records if r.get("type") == "train_point"]
    preds = sorted(
        [r for r in records if r.get("type") == "prediction"],
        key=lambda r: r["x"],
    )

    train_x = np.array([p["x"] for p in train_pts])

    x_slice = [r["x"] for r in preds]
    e_true = [r["true_e"] for r in preds]
    e_pred = [r["gp_mean"] for r in preds]
    e_std = [np.sqrt(max(r["gp_var"], 0.0)) for r in preds]
    dist_to_data = [float(np.min(np.abs(r["x"] - train_x))) for r in preds]
    in_trust = [1.0 if d <= trust_r else 0.0 for d in dist_to_data]

    dst = OUTDIR / "mb_trust.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "slice", {
            "x": x_slice,
            "E_true": e_true,
            "E_pred": e_pred,
            "E_std": e_std,
            "dist_to_data": dist_to_data,
            "in_trust": in_trust,
        })
        h5_write_points(f, "training", {
            "x": train_x.tolist(),
            "y": [y_slice] * len(train_x),
        })
        f.attrs["trust_radius"] = trust_r
        f.attrs["y_slice"] = y_slice
    print(f"  wrote {dst}")


def convert_leps_minimize():
    """leps_minimize_comparison.jsonl -> leps_minimize.h5

    table {oracle_calls, max_fatom, method}
    """
    src = ROOT / "leps_minimize_comparison.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    all_records = read_jsonl(src)
    records = [r for r in all_records if "method" in r and not r.get("summary")]
    summary = next((r for r in all_records if r.get("summary")), {})

    dst = OUTDIR / "leps_minimize.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": [r["oracle_calls"] for r in records],
            "max_fatom": [r.get("max_fatom", 0.0) for r in records],
            "method": [label(r["method"]) for r in records],
        })
        f.attrs["conv_tol"] = summary.get("conv_tol", 0.01)
    print(f"  wrote {dst}")


def convert_petmad_minimize():
    """petmad_minimize_comparison.jsonl -> petmad_minimize.h5

    table {oracle_calls, max_fatom, method}
    """
    src = ROOT / "petmad_minimize_comparison.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    all_records = read_jsonl(src)
    records = [r for r in all_records if "method" in r and not r.get("summary")]
    summary = next((r for r in all_records if r.get("summary")), {})

    dst = OUTDIR / "petmad_minimize.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": [r["oracle_calls"] for r in records],
            "max_fatom": [r.get("max_fatom", 0.0) for r in records],
            "method": [label(r["method"]) for r in records],
        })
        f.attrs["conv_tol"] = summary.get("conv_tol", 0.01)
    print(f"  wrote {dst}")


def convert_leps_neb():
    """leps_neb_comparison.jsonl -> leps_neb.h5 + leps_aie_oie.h5"""
    src = ROOT / "leps_neb_comparison.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    summary = next((r for r in records if r.get("summary")), {})
    grid_meta = None
    grid_data, neb_path, endpoints = [], [], []
    oc, mf, methods = [], [], []

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
        f.attrs["conv_tol"] = summary.get("conv_tol", 0.1)
    print(f"  wrote {dst1}")

    # leps_neb.h5 (grid + path + endpoints)
    if grid_meta and grid_data:
        nx, ny = grid_meta["nx"], grid_meta["ny"]
        rab_range = (grid_meta["rab_min"], grid_meta["rab_max"])
        rbc_range = (grid_meta["rbc_min"], grid_meta["rbc_max"])

        energy = grid_to_array(grid_data, nx, ny, "energy")

        dst2 = OUTDIR / "leps_neb.h5"
        with h5py.File(dst2, "w") as f:
            h5_write_grid(f, "energy", energy, rab_range, rbc_range)

            if neb_path:
                neb_path.sort(key=lambda r: r["image"])
                h5_write_path(f, "neb", {
                    "rAB": [r["rAB"] for r in neb_path],
                    "rBC": [r["rBC"] for r in neb_path],
                })

            # Saddle points (from Rust example output)
            saddles = [r for r in records if r.get("type") == "saddle"]
            if saddles:
                h5_write_points(f, "saddles", {
                    "rAB": [s["rAB"] for s in saddles],
                    "rBC": [s["rBC"] for s in saddles],
                })

            if endpoints:
                h5_write_points(f, "endpoints", {
                    "rAB": [r["rAB"] for r in endpoints],
                    "rBC": [r["rBC"] for r in endpoints],
                })
        print(f"  wrote {dst2}")


def convert_leps_fps():
    """leps_fps.jsonl -> leps_fps.h5

    points/selected {pc1, pc2}, points/pruned {pc1, pc2}
    PCA computed from raw feature vectors.
    """
    src = ROOT / "leps_fps.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)
    candidates = [r for r in records if r.get("type") == "candidate"]
    if not candidates:
        print(f"  skip (no candidate records)", file=sys.stderr)
        return

    features = np.array([c["features"] for c in candidates])  # (n, d)
    selected = np.array([c["selected"] for c in candidates])

    # PCA to 2D
    F = features.T  # (d, n)
    F_c = F - F.mean(axis=1, keepdims=True)
    C = F_c @ F_c.T / F.shape[1]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    proj = eigvecs[:, idx[:2]].T @ F_c  # (2, n)

    sel = proj[:, selected]
    pru = proj[:, ~selected]

    dst = OUTDIR / "leps_fps.h5"
    with h5py.File(dst, "w") as f:
        h5_write_points(f, "selected", {"pc1": sel[0], "pc2": sel[1]})
        h5_write_points(f, "pruned", {"pc1": pru[0], "pc2": pru[1]})
    print(f"  wrote {dst}")


def convert_leps_rff():
    """leps_rff_quality.jsonl -> leps_rff.h5

    table {D_rff, energy_mae_vs_true, gradient_mae_vs_true,
           energy_mae_vs_gp, gradient_mae_vs_gp},
    attrs: gp_e_mae, gp_g_mae
    """
    src = ROOT / "leps_rff_quality.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    exact = next(r for r in records if r.get("type") == "exact_gp")
    rffs = sorted(
        [r for r in records if r.get("type") == "rff"],
        key=lambda r: r["d_rff"],
    )

    dst = OUTDIR / "leps_rff.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "D_rff": [r["d_rff"] for r in rffs],
            "energy_mae_vs_true": [r["energy_mae_vs_true"] for r in rffs],
            "gradient_mae_vs_true": [r["gradient_mae_vs_true"] for r in rffs],
            "energy_mae_vs_gp": [r["energy_mae_vs_gp"] for r in rffs],
            "gradient_mae_vs_gp": [r["gradient_mae_vs_gp"] for r in rffs],
        })
        f.attrs["gp_e_mae"] = exact["energy_mae"]
        f.attrs["gp_g_mae"] = exact["gradient_mae"]
    print(f"  wrote {dst}")


def convert_petmad_rff():
    """petmad_rff_quality.jsonl -> petmad_rff.h5

    Same format as leps_rff.h5 but with system metadata.
    """
    src = ROOT / "petmad_rff_quality.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)

    exact = next(r for r in records if r.get("type") == "exact_gp")
    rffs = sorted(
        [r for r in records if r.get("type") == "rff"],
        key=lambda r: r["d_rff"],
    )

    dst = OUTDIR / "petmad_rff.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "D_rff": [r["d_rff"] for r in rffs],
            "energy_mae_vs_true": [r["energy_mae_vs_true"] for r in rffs],
            "gradient_mae_vs_true": [r["gradient_mae_vs_true"] for r in rffs],
            "energy_mae_vs_gp": [r["energy_mae_vs_gp"] for r in rffs],
            "gradient_mae_vs_gp": [r["gradient_mae_vs_gp"] for r in rffs],
        })
        f.attrs["gp_e_mae"] = exact["energy_mae"]
        f.attrs["gp_g_mae"] = exact["gradient_mae"]
        f.attrs["system"] = exact.get("system", "C2H4NO")
        f.attrs["n_atoms"] = exact.get("n_atoms", 9)
        f.attrs["n_features"] = exact.get("n_features", 36)
    print(f"  wrote {dst}")


def convert_leps_nll():
    """leps_nll_landscape.jsonl -> leps_nll.h5

    grids/nll, grids/grad_norm (2D grids in log_sigma2 x log_theta),
    points/optimum {log_sigma2, log_theta}
    """
    src = ROOT / "leps_nll_landscape.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = read_jsonl(src)
    if not records:
        print("  skip (no records)", file=sys.stderr)
        return

    # Separate SCG optimum record from grid data
    scg_opt = next((r for r in records if r.get("type") == "scg_optimum"), None)
    grid_recs = [r for r in records if "nll" in r]

    ls2_vals = sorted(set(r["log_sigma2"] for r in grid_recs))
    lt_vals = sorted(set(r["log_theta"] for r in grid_recs))
    nx, ny = len(ls2_vals), len(lt_vals)

    ls2_idx = {v: i for i, v in enumerate(ls2_vals)}
    lt_idx = {v: i for i, v in enumerate(lt_vals)}

    nll_grid = np.full((nx, ny), np.nan)
    grad_grid = np.full((nx, ny), np.nan)
    for r in grid_recs:
        ix = ls2_idx[r["log_sigma2"]]
        iy = lt_idx[r["log_theta"]]
        nll_grid[ix, iy] = r["nll"]
        grad_grid[ix, iy] = r["grad_norm"]

    ls2_range = (ls2_vals[0], ls2_vals[-1])
    lt_range = (lt_vals[0], lt_vals[-1])

    # Use SCG optimum if available, otherwise fall back to grid minimum
    if scg_opt:
        opt_ls2 = scg_opt["log_sigma2"]
        opt_lt = scg_opt["log_theta"]
    else:
        finite_mask = np.isfinite(nll_grid)
        if np.any(finite_mask):
            min_idx = np.unravel_index(np.nanargmin(nll_grid), nll_grid.shape)
            opt_ls2 = ls2_vals[min_idx[0]]
            opt_lt = lt_vals[min_idx[1]]
        else:
            opt_ls2, opt_lt = 0.0, 0.0

    dst = OUTDIR / "leps_nll.h5"
    with h5py.File(dst, "w") as f:
        h5_write_grid(f, "nll", nll_grid, ls2_range, lt_range)
        h5_write_grid(f, "grad_norm", grad_grid, ls2_range, lt_range)
        h5_write_points(f, "optimum", {
            "log_sigma2": [opt_ls2],
            "log_theta": [opt_lt],
        })
    print(f"  wrote {dst}")


def convert_rpc_dimer():
    """rpc_dimer.jsonl -> rpc_dimer.h5

    table {oracle_calls, force_norm, method}
    Molecular dimer saddle search (d_000 C3H5 via PET-MAD).
    """
    src = ROOT / "rpc_dimer.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    all_records = read_jsonl(src)
    records = [r for r in all_records if "method" in r and not r.get("summary")]
    summary = next((r for r in all_records if r.get("summary")), {})

    dst = OUTDIR / "rpc_dimer.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "oracle_calls": [r["oracle_calls"] for r in records],
            "force_norm": [r["force"] for r in records],
            "method": [label(r["method"]) for r in records],
        })
        f.attrs["conv_tol"] = summary.get("conv_tol", 0.01)
    print(f"  wrote {dst}")


def convert_hcn_convergence():
    """hcn_neb_comparison.jsonl -> hcn_convergence.h5

    table {oracle_calls, max_force, method}
    """
    src = ROOT / "hcn_neb_comparison.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    all_records = read_jsonl(src)
    records = [
        r for r in all_records
        if "method" in r and not r.get("summary") and not r.get("type")
    ]
    summary = next((r for r in all_records if r.get("summary")), {})
    if not records:
        print("  skip (no convergence records)", file=sys.stderr)
        return

    dst = OUTDIR / "hcn_convergence.h5"
    with h5py.File(dst, "w") as f:
        cols = {
            "oracle_calls": [r["oracle_calls"] for r in records],
            "max_force": [r["max_force"] for r in records],
            "method": [label(r["method"]) for r in records],
        }
        # Include ci_force if available (for CI-NEB convergence metric)
        if "ci_force" in records[0]:
            cols["ci_force"] = [r.get("ci_force", r["max_force"]) for r in records]
        h5_write_table(f, "table", cols)
        f.attrs["conv_tol"] = summary.get("conv_tol", 0.15)
    print(f"  wrote {dst}")


def convert_hcn_neb_profile():
    """hcn_neb_comparison.jsonl -> hcn_neb.h5

    table {image, energy, method}
    """
    src = ROOT / "hcn_neb_comparison.jsonl"
    if not src.exists():
        print(f"  skip (no {src.name})", file=sys.stderr)
        return

    records = [r for r in read_jsonl(src) if r.get("type") == "path_energy"]
    if not records:
        print("  skip (no path_energy records)", file=sys.stderr)
        return

    # Compute relative energies (relative to first image of each method)
    from itertools import groupby as igroupby
    keyfn = lambda r: r.get("method", "gp_neb_oie")
    images, energies, methods = [], [], []
    for method_key, grp in igroupby(records, key=keyfn):
        pts = list(grp)
        e_ref = pts[0]["energy"]
        for r in pts:
            images.append(r["image"])
            energies.append(r["energy"] - e_ref)
            methods.append(label(method_key))

    dst = OUTDIR / "hcn_neb.h5"
    with h5py.File(dst, "w") as f:
        h5_write_table(f, "table", {
            "image": images,
            "energy": energies,
            "method": methods,
        })
    print(f"  wrote {dst}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Converting JSONL -> HDF5 for Julia plotters...")

    # Muller-Brown
    print("mb_gp + mb_variance:")
    convert_mb_gp_quality()
    print("mb_gp_scattered:")
    convert_mb_gp_scattered()
    print("mb_neb:")
    convert_mb_neb()
    print("mb_dimer:")
    convert_mb_dimer()
    print("mb_hyperparams:")
    convert_mb_hyperparams()
    print("mb_trust:")
    convert_mb_trust()

    # LEPS
    print("leps_minimize:")
    convert_leps_minimize()
    print("leps_neb + leps_aie_oie:")
    convert_leps_neb()
    print("leps_fps:")
    convert_leps_fps()
    print("leps_rff:")
    convert_leps_rff()
    print("leps_nll:")
    convert_leps_nll()

    print("petmad_minimize:")
    convert_petmad_minimize()
    print("petmad_rff:")
    convert_petmad_rff()

    # RPC dimer (molecular saddle search)
    print("rpc_dimer:")
    convert_rpc_dimer()

    # HCN (partial if run was incomplete)
    print("hcn_convergence:")
    convert_hcn_convergence()
    print("hcn_neb:")
    convert_hcn_neb_profile()

    print("\nDone.")


if __name__ == "__main__":
    main()
