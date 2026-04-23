//! Oracle wrappers for potential evaluation.
//!
//! Backends:
//! - `rgpot`: remote RPC via the Rust `rgpot-core` crate
//! - `rgpot_local`: direct local metatomic evaluation via the `rgpot` C API

#[cfg(all(feature = "rgpot", feature = "rgpot_local"))]
compile_error!("Enable at most one rgpot backend feature: 'rgpot' or 'rgpot_local'");

#[cfg(feature = "rgpot")]
use rgpot_core::c_api::types::{
    rgpot_force_input_create, rgpot_force_input_free, rgpot_force_out_create,
};
#[cfg(feature = "rgpot")]
use rgpot_core::potential::rgpot_potential_t;
#[cfg(feature = "rgpot")]
use rgpot_core::rpc::client::RpcClient;
#[cfg(feature = "rgpot")]
use rgpot_core::status::rgpot_status_t;
#[cfg(feature = "rgpot")]
use rgpot_core::tensor::rgpot_tensor_free;

#[cfg(feature = "rgpot_local")]
use dlpk::sys::DLManagedTensorVersioned;

#[cfg(feature = "rgpot")]
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

#[cfg(feature = "rgpot")]
unsafe impl Send for RgpotOracle {}

#[cfg(feature = "rgpot")]
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

            let forces_ptr = output.forces;
            if forces_ptr.is_null() {
                return Err("rgpot returned null forces".to_string());
            }
            let dl = &*forces_ptr;
            let data = dl.dl_tensor.data as *const f64;
            let forces_slice = std::slice::from_raw_parts(data, n_atoms * 3);
            let gradient: Vec<f64> = forces_slice.iter().map(|&f| -f).collect();

            rgpot_tensor_free(forces_ptr);

            Ok((energy, gradient))
        }
    }
}

#[cfg(feature = "rgpot")]
/// Oracle that connects to a remote eOn serve instance via Cap'n Proto RPC.
pub struct RpcOracle {
    client: RpcClient,
    atomic_numbers: Vec<i32>,
    box_matrix: [f64; 9],
}

#[cfg(feature = "rgpot")]
impl RpcOracle {
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
            let gradient: Vec<f64> = forces_slice.iter().map(|&f| -f).collect();

            rgpot_tensor_free(forces_ptr);

            Ok((energy, gradient))
        }
    }

    pub fn as_oracle_fn(&mut self) -> impl FnMut(&[f64]) -> (f64, Vec<f64>) + '_ {
        move |x: &[f64]| {
            self.evaluate(x)
                .unwrap_or_else(|e| panic!("RPC oracle evaluation failed: {}", e))
        }
    }
}

#[cfg(feature = "rgpot_local")]
#[repr(C)]
struct rgpot_potential_t {
    _private: [u8; 0],
}

#[cfg(feature = "rgpot_local")]
#[repr(C)]
struct rgpot_force_input_t {
    positions: *mut DLManagedTensorVersioned,
    atomic_numbers: *mut DLManagedTensorVersioned,
    box_matrix: *mut DLManagedTensorVersioned,
}

#[cfg(feature = "rgpot_local")]
#[repr(C)]
struct rgpot_force_out_t {
    forces: *mut DLManagedTensorVersioned,
    energy: f64,
    variance: f64,
}

#[cfg(feature = "rgpot_local")]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
enum rgpot_status_t {
    RGPOT_SUCCESS = 0,
    RGPOT_INVALID_PARAMETER = 1,
    RGPOT_INTERNAL_ERROR = 2,
    RGPOT_RPC_ERROR = 3,
    RGPOT_BUFFER_SIZE_ERROR = 4,
}

#[cfg(feature = "rgpot_local")]
#[repr(C)]
struct rgpot_metatomic_config_t {
    model_path: *const std::os::raw::c_char,
    device: *const std::os::raw::c_char,
    length_unit: *const std::os::raw::c_char,
    extensions_directory: *const std::os::raw::c_char,
    check_consistency: bool,
    uncertainty_threshold: f64,
    dtype_override: *const std::os::raw::c_char,
}

#[cfg(feature = "rgpot_local")]
unsafe extern "C" {
    fn rgpot_force_input_create(
        n_atoms: usize,
        pos: *mut f64,
        atmnrs: *mut i32,
        box_: *mut f64,
    ) -> rgpot_force_input_t;
    fn rgpot_force_input_free(input: *mut rgpot_force_input_t);
    fn rgpot_force_out_create() -> rgpot_force_out_t;
    fn rgpot_tensor_free(tensor: *mut DLManagedTensorVersioned);
    fn rgpot_potential_calculate(
        pot: *const rgpot_potential_t,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t;
    fn rgpot_potential_free(pot: *mut rgpot_potential_t);
    fn rgpot_metatomic_potential_new(
        config: *const rgpot_metatomic_config_t,
    ) -> *mut rgpot_potential_t;
}

#[cfg(feature = "rgpot_local")]
#[derive(Debug, Clone)]
pub struct LocalMetatomicConfig {
    pub model_path: std::path::PathBuf,
    pub device: String,
    pub length_unit: String,
    pub extensions_directory: Option<std::path::PathBuf>,
    pub check_consistency: bool,
    pub uncertainty_threshold: f64,
    pub dtype_override: Option<String>,
}

#[cfg(feature = "rgpot_local")]
impl Default for LocalMetatomicConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/pet-mad-xs-v1.5.0.pt"),
            device: "cpu".to_string(),
            length_unit: "angstrom".to_string(),
            extensions_directory: None,
            check_consistency: false,
            uncertainty_threshold: -1.0,
            dtype_override: None,
        }
    }
}

#[cfg(feature = "rgpot_local")]
pub struct LocalMetatomicOracle {
    pot: *mut rgpot_potential_t,
    atomic_numbers: Vec<i32>,
    box_matrix: [f64; 9],
}

#[cfg(feature = "rgpot_local")]
unsafe impl Send for LocalMetatomicOracle {}

#[cfg(feature = "rgpot_local")]
impl LocalMetatomicOracle {
    pub fn new(
        config: &LocalMetatomicConfig,
        atomic_numbers: Vec<i32>,
        box_matrix: [f64; 9],
    ) -> Result<Self, String> {
        use std::ffi::CString;

        let model_path = CString::new(config.model_path.to_string_lossy().into_owned())
            .map_err(|e| format!("Invalid model path: {e}"))?;
        let device = CString::new(config.device.clone())
            .map_err(|e| format!("Invalid device string: {e}"))?;
        let length_unit = CString::new(config.length_unit.clone())
            .map_err(|e| format!("Invalid length unit: {e}"))?;
        let extensions_directory = CString::new(
            config
                .extensions_directory
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
        )
        .map_err(|e| format!("Invalid extensions directory: {e}"))?;
        let dtype_override = CString::new(config.dtype_override.clone().unwrap_or_default())
            .map_err(|e| format!("Invalid dtype override: {e}"))?;

        let ffi_config = rgpot_metatomic_config_t {
            model_path: model_path.as_ptr(),
            device: device.as_ptr(),
            length_unit: length_unit.as_ptr(),
            extensions_directory: extensions_directory.as_ptr(),
            check_consistency: config.check_consistency,
            uncertainty_threshold: config.uncertainty_threshold,
            dtype_override: dtype_override.as_ptr(),
        };

        let pot = unsafe { rgpot_metatomic_potential_new(&ffi_config) };
        if pot.is_null() {
            return Err("Failed to create local rgpot metatomic potential".to_string());
        }

        Ok(Self {
            pot,
            atomic_numbers,
            box_matrix,
        })
    }

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

            let status = rgpot_potential_calculate(self.pot, &input, &mut output);

            rgpot_force_input_free(&mut input);

            if status != rgpot_status_t::RGPOT_SUCCESS {
                return Err(format!(
                    "local rgpot calculation failed with status {:?}",
                    status
                ));
            }

            if output.forces.is_null() {
                return Err("local rgpot returned null forces".to_string());
            }

            let dl = &*output.forces;
            let data = dl.dl_tensor.data as *const f64;
            let forces_slice = std::slice::from_raw_parts(data, n_atoms * 3);
            let gradient: Vec<f64> = forces_slice.iter().map(|&f| -f).collect();
            let energy = output.energy;

            rgpot_tensor_free(output.forces);

            Ok((energy, gradient))
        }
    }
}

#[cfg(feature = "rgpot_local")]
impl Drop for LocalMetatomicOracle {
    fn drop(&mut self) {
        unsafe {
            if !self.pot.is_null() {
                rgpot_potential_free(self.pot);
            }
        }
    }
}
