//! I/O for molecular structures.
//!
//! Feature-gated behind `io`. Provides:
//! - ExtXYZ reading/writing via chemfiles
//! - CON file reading via readcon-core
//! - Unified `read_structure` dispatcher

use chemfiles::{Frame, Trajectory};
use readcon_core::helpers::symbol_to_atomic_number;
use readcon_core::iterators::ConFrameIterator;
use readcon_core::types::ConFrameBuilder;
use readcon_core::writer::ConFrameWriter;

/// A molecular configuration.
#[derive(Debug, Clone)]
pub struct MolConfig {
    /// Flat Cartesian coordinates [x1,y1,z1, x2,y2,z2, ...].
    pub positions: Vec<f64>,
    /// Atomic numbers for each atom.
    pub atomic_numbers: Vec<i32>,
    /// Energy (if present in frame properties).
    pub energy: Option<f64>,
    /// Flat forces [fx1,fy1,fz1, ...] (if present).
    pub forces: Option<Vec<f64>>,
    /// Cell matrix (row-major 3x3, zeros for non-periodic).
    pub cell: [[f64; 3]; 3],
}

// ---------------------------------------------------------------------------
// Unified dispatcher
// ---------------------------------------------------------------------------

/// Read the first structure from a file, dispatching on extension.
///
/// Supported formats:
/// - `.extxyz`, `.xyz` -> chemfiles
/// - `.con`, `.convel` -> readcon-core
pub fn read_structure(path: &str) -> Result<MolConfig, String> {
    let lower = path.to_lowercase();
    if lower.ends_with(".con") || lower.ends_with(".convel") {
        let frames = read_con(path)?;
        frames
            .into_iter()
            .next()
            .ok_or_else(|| format!("No frames in {}", path))
    } else if lower.ends_with(".extxyz") || lower.ends_with(".xyz") {
        let frames = read_extxyz(path)?;
        frames
            .into_iter()
            .next()
            .ok_or_else(|| format!("No frames in {}", path))
    } else {
        Err(format!(
            "Unsupported file format for '{}' (supported: .extxyz, .xyz, .con, .convel)",
            path
        ))
    }
}

// ---------------------------------------------------------------------------
// CON reader via readcon-core
// ---------------------------------------------------------------------------

/// Read all frames from a CON file.
pub fn read_con(path: &str) -> Result<Vec<MolConfig>, String> {
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let iter = ConFrameIterator::new(&contents);
    let mut configs = Vec::new();

    for result in iter {
        let frame = result.map_err(|e| format!("CON parse error in {}: {:?}", path, e))?;
        let n = frame.atom_data.len();

        let mut positions = Vec::with_capacity(3 * n);
        let mut atomic_numbers = Vec::with_capacity(n);

        for atom in &frame.atom_data {
            positions.push(atom.x);
            positions.push(atom.y);
            positions.push(atom.z);
            let z = symbol_to_atomic_number(&atom.symbol);
            atomic_numbers.push(z as i32);
        }

        // Build row-major 3x3 cell from box lengths + angles
        let [a, b, c] = frame.header.boxl;
        let [alpha, beta, gamma] = frame.header.angles;

        let cell = if a == 0.0 && b == 0.0 && c == 0.0 {
            // Non-periodic default
            [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
        } else {
            let ar = alpha.to_radians();
            let br = beta.to_radians();
            let gr = gamma.to_radians();
            let cos_g = gr.cos();
            let sin_g = gr.sin();
            let cos_b = br.cos();
            let cos_a = ar.cos();

            let bx = b * cos_g;
            let by = b * sin_g;
            let cx = c * cos_b;
            let cy = c * (cos_a - cos_b * cos_g) / sin_g;
            let cz = (c * c - cx * cx - cy * cy).max(0.0).sqrt();

            [[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]]
        };

        configs.push(MolConfig {
            positions,
            atomic_numbers,
            energy: None,
            forces: None,
            cell,
        });
    }

    Ok(configs)
}

// ---------------------------------------------------------------------------
// ExtXYZ via chemfiles
// ---------------------------------------------------------------------------

/// Read all frames from an extxyz file.
pub fn read_extxyz(path: &str) -> Result<Vec<MolConfig>, String> {
    // Always use XYZ format hint since chemfiles may not recognize .extxyz
    // and ORCA-style XYZ files can cause issues with auto-detection.
    let mut traj = Trajectory::open_with_format(path, 'r', "XYZ")
        .map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let nsteps = traj.nsteps();
    let mut configs = Vec::with_capacity(nsteps as usize);
    let mut frame = Frame::new();

    for step in 0..nsteps {
        traj.read_step(step, &mut frame)
            .map_err(|e| format!("Failed to read step {}: {}", step, e))?;

        let natoms = frame.size();
        let pos = frame.positions();
        let mut positions = Vec::with_capacity(natoms * 3);
        for p in pos {
            positions.push(p[0]);
            positions.push(p[1]);
            positions.push(p[2]);
        }

        let mut atomic_numbers = Vec::with_capacity(natoms);
        for atom in frame.iter_atoms() {
            atomic_numbers.push(atom.atomic_number() as i32);
        }

        let energy = frame
            .get("energy")
            .and_then(|p| match p {
                chemfiles::Property::Double(v) => Some(v),
                _ => None,
            })
            .or_else(|| {
                frame.get("Energy").and_then(|p| match p {
                    chemfiles::Property::Double(v) => Some(v),
                    _ => None,
                })
            });

        let forces = {
            let mut fvec = Vec::with_capacity(natoms * 3);
            let mut found = false;
            for i in 0..natoms {
                let atom = frame.atom(i);
                if let Some(chemfiles::Property::Vector3D(f)) = atom.get("forces") {
                    fvec.push(f[0]);
                    fvec.push(f[1]);
                    fvec.push(f[2]);
                    found = true;
                } else if let Some(chemfiles::Property::Vector3D(f)) = atom.get("force") {
                    fvec.push(f[0]);
                    fvec.push(f[1]);
                    fvec.push(f[2]);
                    found = true;
                } else {
                    fvec.push(0.0);
                    fvec.push(0.0);
                    fvec.push(0.0);
                }
            }
            if found { Some(fvec) } else { None }
        };

        let cell = frame.cell().matrix();

        configs.push(MolConfig {
            positions,
            atomic_numbers,
            energy,
            forces,
            cell,
        });
    }

    Ok(configs)
}

/// Write a sequence of configurations to an extxyz file.
pub fn write_extxyz(path: &str, configs: &[MolConfig]) -> Result<(), String> {
    let mut traj =
        Trajectory::open(path, 'w').map_err(|e| format!("Failed to open {}: {}", path, e))?;

    for (step, cfg) in configs.iter().enumerate() {
        let natoms = cfg.atomic_numbers.len();
        let mut frame = Frame::new();
        frame.resize(natoms);

        let pos_mut = frame.positions_mut();
        for i in 0..natoms {
            pos_mut[i] = [
                cfg.positions[3 * i],
                cfg.positions[3 * i + 1],
                cfg.positions[3 * i + 2],
            ];
        }

        for i in 0..natoms {
            let sym = element_symbol(cfg.atomic_numbers[i]);
            frame.atom_mut(i).set_name(sym);
        }

        if let Some(e) = cfg.energy {
            frame.set("energy", e);
        }

        frame.set_step(step);

        if let Some(ref forces) = cfg.forces {
            for i in 0..natoms {
                let mut atom = frame.atom_mut(i);
                atom.set(
                    "forces",
                    chemfiles::Property::Vector3D([
                        forces[3 * i],
                        forces[3 * i + 1],
                        forces[3 * i + 2],
                    ]),
                );
            }
        }

        traj.write(&frame)
            .map_err(|e| format!("Failed to write step {}: {}", step, e))?;
    }

    Ok(())
}

/// Write a sequence of configurations to a CON file via readcon-core.
pub fn write_con(path: &str, configs: &[MolConfig]) -> Result<(), String> {
    let mut writer = ConFrameWriter::from_path(path)
        .map_err(|e| format!("Failed to create {}: {}", path, e))?;

    for cfg in configs {
        let natoms = cfg.atomic_numbers.len();
        let boxl = [cfg.cell[0][0], cfg.cell[1][1], cfg.cell[2][2]];
        let angles = [90.0, 90.0, 90.0]; // orthorhombic assumption

        let mut builder = ConFrameBuilder::new(boxl, angles)
            .prebox_header([
                "Generated by ChemGP".to_string(),
                String::new(),
            ]);

        for i in 0..natoms {
            let z = cfg.atomic_numbers[i];
            let sym = element_symbol(z);
            let mass = element_mass(z);
            builder.add_atom(
                sym,
                cfg.positions[3 * i],
                cfg.positions[3 * i + 1],
                cfg.positions[3 * i + 2],
                false,
                i as u64,
                mass,
            );
        }

        writer
            .write_frame(&builder.build())
            .map_err(|e| format!("Failed to write frame to {}: {}", path, e))?;
    }
    Ok(())
}

/// Write a NEB .dat file (eOn-compatible format for rgpycrumbs).
///
/// Columns: img, rxn_coord, energy, f_para
pub fn write_neb_dat(
    path: &str,
    images: &[Vec<f64>],
    energies: &[f64],
    gradients: &[Vec<f64>],
) -> Result<(), String> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let f = File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;
    let mut w = BufWriter::new(f);

    writeln!(w, "img\trxn_coord\tenergy\tf_para")
        .map_err(|e| format!("Write error: {}", e))?;

    let n = images.len();
    let e_ref = energies[0];

    // Cumulative reaction coordinate (Cartesian distance)
    let mut rxn_coord = vec![0.0f64; n];
    for i in 1..n {
        let d: f64 = images[i]
            .iter()
            .zip(images[i - 1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        rxn_coord[i] = rxn_coord[i - 1] + d;
    }

    for i in 0..n {
        // Parallel force: project gradient onto path tangent
        let f_para = if i == 0 || i == n - 1 {
            0.0
        } else {
            let tau: Vec<f64> = images[i + 1]
                .iter()
                .zip(images[i - 1].iter())
                .map(|(a, b)| a - b)
                .collect();
            let tau_norm = tau.iter().map(|v| v * v).sum::<f64>().sqrt();
            if tau_norm > 1e-12 {
                let g_dot_tau: f64 = gradients[i]
                    .iter()
                    .zip(tau.iter())
                    .map(|(g, t)| g * t)
                    .sum::<f64>();
                -g_dot_tau / tau_norm
            } else {
                0.0
            }
        };

        writeln!(
            w,
            "{}\t{:.6}\t{:.6}\t{:.6}",
            i,
            rxn_coord[i],
            energies[i] - e_ref,
            f_para
        )
        .map_err(|e| format!("Write error: {}", e))?;
    }

    Ok(())
}

fn element_mass(z: i32) -> f64 {
    match z {
        1 => 1.008,
        2 => 4.003,
        6 => 12.011,
        7 => 14.007,
        8 => 15.999,
        9 => 18.998,
        14 => 28.086,
        15 => 30.974,
        16 => 32.065,
        17 => 35.453,
        26 => 55.845,
        29 => 63.546,
        78 => 195.078,
        79 => 196.967,
        _ => 0.0,
    }
}

fn element_symbol(z: i32) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        26 => "Fe",
        29 => "Cu",
        30 => "Zn",
        47 => "Ag",
        78 => "Pt",
        79 => "Au",
        _ => "X",
    }
}
