//! Oracle wrapper around rgpot-core for potential evaluation.
//!
//! Feature-gated behind `rgpot`. Provides a safe Rust interface
//! that wraps rgpot-core's C-ABI callback-based potentials.

use rgpot_core::{
    rgpot_force_input_create, rgpot_force_out_create, rgpot_force_input_free,
    rgpot_potential_calculate, rgpot_potential_t, rgpot_status_t,
};

/// Safe wrapper around an rgpot potential handle.
///
/// The caller is responsible for the lifetime of the underlying
/// `rgpot_potential_t` (typically created via rgpot C API or
/// a language-specific binding).
pub struct RgpotOracle {
    /// Raw pointer to the rgpot potential. NOT owned.
    pot: *const rgpot_potential_t,
    /// Atomic numbers for the system (fixed topology).
    atomic_numbers: Vec<i32>,
    /// Box matrix (row-major 3x3). Zeros for non-periodic.
    box_matrix: [f64; 9],
}

// Safety: the rgpot_potential_t is thread-safe (stateless callback dispatch).
unsafe impl Send for RgpotOracle {}

impl RgpotOracle {
    /// Create an oracle wrapper from an existing rgpot potential handle.
    ///
    /// # Safety
    /// `pot` must point to a valid, live `rgpot_potential_t` that outlives
    /// this `RgpotOracle`.
    pub unsafe fn new(
        pot: *const rgpot_potential_t,
        atomic_numbers: Vec<i32>,
        box_matrix: [f64; 9],
    ) -> Self {
        Self {
            pot,
            atomic_numbers,
            box_matrix,
        }
    }

    /// Evaluate energy and forces for a flat coordinate vector.
    ///
    /// `positions`: flat [x1,y1,z1, x2,y2,z2, ...] in Angstroms.
    /// Returns (energy, gradient) where gradient = -forces.
    pub fn evaluate(&self, positions: &[f64]) -> Result<(f64, Vec<f64>), String> {
        let n_atoms = self.atomic_numbers.len();
        assert_eq!(
            positions.len(),
            n_atoms * 3,
            "Position vector length mismatch"
        );

        let mut pos = positions.to_vec();
        let mut atnrs = self.atomic_numbers.clone();
        let mut box_mat = self.box_matrix;

        unsafe {
            let input = rgpot_force_input_create(
                n_atoms,
                pos.as_mut_ptr(),
                atnrs.as_mut_ptr(),
                box_mat.as_mut_ptr(),
            );
            let mut output = rgpot_force_out_create();

            let status = rgpot_potential_calculate(self.pot, &input, &mut output);

            // Clean up input metadata (does NOT free our data arrays)
            let mut input_mut = input;
            rgpot_force_input_free(&mut input_mut);

            if status != rgpot_status_t::RGPOT_SUCCESS {
                return Err(format!("rgpot calculation failed: {:?}", status));
            }

            let energy = output.energy;

            // Extract forces from DLPack tensor
            // Forces tensor is [n_atoms, 3], row-major f64
            let forces_ptr = output.forces;
            if forces_ptr.is_null() {
                return Err("rgpot returned null forces".to_string());
            }
            let dl = &*forces_ptr;
            let data = dl.dl_tensor.data as *const f64;
            let forces_slice = std::slice::from_raw_parts(data, n_atoms * 3);

            // Gradient = -forces (convention: force = -dE/dx)
            let gradient: Vec<f64> = forces_slice.iter().map(|&f| -f).collect();

            // Free forces tensor
            if let Some(deleter) = dl.deleter {
                deleter(forces_ptr as *mut _);
            }

            Ok((energy, gradient))
        }
    }

    /// Convenience: return as a closure matching the OracleFn signature.
    pub fn as_oracle_fn(&self) -> impl Fn(&[f64]) -> (f64, Vec<f64>) + '_ {
        move |x: &[f64]| {
            self.evaluate(x)
                .unwrap_or_else(|e| panic!("Oracle evaluation failed: {}", e))
        }
    }
}
