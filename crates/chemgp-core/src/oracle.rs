//! Oracle wrappers for potential evaluation.
//!
//! Feature-gated behind `rgpot`. Provides:
//! - [`RgpotOracle`]: wraps a callback-based `rgpot_potential_t` handle (local)
//! - [`RpcOracle`]: connects to a remote eOn serve instance via Cap'n Proto RPC

use rgpot_core::c_api::types::{
    rgpot_force_input_create, rgpot_force_input_free, rgpot_force_out_create,
};
use rgpot_core::potential::rgpot_potential_t;
use rgpot_core::rpc::client::RpcClient;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_free;

// ---------------------------------------------------------------------------
// RgpotOracle: local callback-based potential
// ---------------------------------------------------------------------------

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
            let mut input = rgpot_force_input_create(
                n_atoms,
                pos.as_mut_ptr(),
                atnrs.as_mut_ptr(),
                box_mat.as_mut_ptr(),
            );
            let mut output = rgpot_force_out_create();

            let status = (*self.pot).calculate(&input, &mut output);

            rgpot_force_input_free(&mut input);

            if status != rgpot_status_t::RGPOT_SUCCESS {
                return Err(format!("rgpot calculation failed: {:?}", status));
            }

            let energy = output.energy;

            // Extract forces from DLPack tensor
            let forces_ptr = output.forces;
            if forces_ptr.is_null() {
                return Err("rgpot returned null forces".to_string());
            }
            let dl = &*forces_ptr;
            let data = dl.dl_tensor.data as *const f64;
            let forces_slice = std::slice::from_raw_parts(data, n_atoms * 3);

            // Gradient = -forces (convention: force = -dE/dx)
            let gradient: Vec<f64> = forces_slice.iter().map(|&f| -f).collect();

            rgpot_tensor_free(forces_ptr);

            Ok((energy, gradient))
        }
    }
}

// ---------------------------------------------------------------------------
// RpcOracle: remote potential via Cap'n Proto RPC
// ---------------------------------------------------------------------------

/// Oracle that connects to a remote eOn serve instance via Cap'n Proto RPC.
///
/// Uses rgpot-core's RPC client to send atomic configurations and receive
/// energies and forces over the network.
pub struct RpcOracle {
    client: RpcClient,
    /// Atomic numbers for the system (fixed topology).
    atomic_numbers: Vec<i32>,
    /// Box matrix (row-major 3x3). Zeros for non-periodic.
    box_matrix: [f64; 9],
}

impl RpcOracle {
    /// Connect to an eOn serve instance at `host:port`.
    ///
    /// `atomic_numbers`: fixed topology (e.g. [6, 7, 1] for C, N, H).
    /// `box_matrix`: row-major 3x3 cell matrix (zeros for non-periodic).
    pub fn new(
        host: &str,
        port: u16,
        atomic_numbers: Vec<i32>,
        box_matrix: [f64; 9],
    ) -> Result<Self, String> {
        let client = RpcClient::new(host, port)?;
        Ok(Self {
            client,
            atomic_numbers,
            box_matrix,
        })
    }

    /// Evaluate energy and forces for a flat coordinate vector.
    ///
    /// `positions`: flat [x1,y1,z1, x2,y2,z2, ...] in Angstroms.
    /// Returns (energy, gradient) where gradient = -forces.
    pub fn evaluate(&mut self, positions: &[f64]) -> Result<(f64, Vec<f64>), String> {
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
            let mut input = rgpot_force_input_create(
                n_atoms,
                pos.as_mut_ptr(),
                atnrs.as_mut_ptr(),
                box_mat.as_mut_ptr(),
            );
            let mut output = rgpot_force_out_create();

            self.client.calculate(&input, &mut output).map_err(|e| {
                rgpot_force_input_free(&mut input);
                format!("RPC calculation failed: {e}")
            })?;

            rgpot_force_input_free(&mut input);

            let energy = output.energy;

            let forces_ptr = output.forces;
            if forces_ptr.is_null() {
                return Err("RPC returned null forces".to_string());
            }
            let dl = &*forces_ptr;
            let data = dl.dl_tensor.data as *const f64;
            let forces_slice = std::slice::from_raw_parts(data, n_atoms * 3);

            // Gradient = -forces
            let gradient: Vec<f64> = forces_slice.iter().map(|&f| -f).collect();

            rgpot_tensor_free(forces_ptr);

            Ok((energy, gradient))
        }
    }

    /// Convenience: return as a closure matching the OracleFn signature.
    ///
    /// Note: requires `&mut self` due to RPC client state.
    pub fn as_oracle_fn(&mut self) -> impl FnMut(&[f64]) -> (f64, Vec<f64>) + '_ {
        move |x: &[f64]| {
            self.evaluate(x)
                .unwrap_or_else(|e| panic!("RPC oracle evaluation failed: {}", e))
        }
    }
}
