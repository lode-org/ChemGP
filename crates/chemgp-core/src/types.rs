//! Core types: TrainingData and GPModel.
//!
//! Ports `types.jl`.

use crate::error::{validate, GpError, GpResult};
use crate::kernel::Kernel;

/// GP noise parameters (energy noise, gradient noise, jitter).
#[derive(Debug, Clone, Copy)]
pub struct GPNoiseParams {
    pub noise_e: f64,
    pub noise_g: f64,
    pub jitter: f64,
}

/// Container for the growing dataset of oracle evaluations.
///
/// X is stored column-major: each column is a D-dimensional configuration.
/// Energies and gradients grow as new points are added.
#[derive(Debug, Clone)]
pub struct TrainingData {
    /// Configuration matrix stored as flat column-major: length D*N.
    /// Column i is data[i*D..(i+1)*D].
    pub data: Vec<f64>,
    pub dim: usize,
    pub energies: Vec<f64>,
    /// Concatenated gradients: length D*N.
    pub gradients: Vec<f64>,
}

impl TrainingData {
    /// Create new empty TrainingData with dimension check.
    ///
    /// # Panics
    ///
    /// Panics if dim is 0 (invalid dimension).
    pub fn new(dim: usize) -> Self {
        if dim == 0 {
            panic!("TrainingData::new: dimension cannot be 0");
        }
        Self {
            data: Vec::new(),
            dim,
            energies: Vec::new(),
            gradients: Vec::new(),
        }
    }

    pub fn npoints(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    /// Add a single oracle evaluation with validation.
    ///
    /// # Errors
    ///
    /// Returns `Err` if x or gradient dimensions don't match,
    /// or if any values are non-finite.
    pub fn add_point(&mut self, x: &[f64], energy: f64, gradient: &[f64]) -> GpResult<()> {
        validate::validate_dimension(self.dim, x.len())?;
        validate::validate_dimension(self.dim, gradient.len())?;
        validate::validate_finite_slice(x)?;
        validate::validate_finite_slice(gradient)?;
        if !energy.is_finite() {
            return Err(GpError::NonFiniteEnergy {
                index: self.npoints(),
                value: energy,
            });
        }
        self.data.extend_from_slice(x);
        self.energies.push(energy);
        self.gradients.extend_from_slice(gradient);
        Ok(())
    }

    /// Get column i (the i-th configuration).
    pub fn col(&self, i: usize) -> &[f64] {
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Normalize targets for GP training.
    /// Returns (y_full, y_mean, y_std) where y_full = [norm_energies; norm_gradients].
    pub fn normalize(&self) -> (Vec<f64>, f64, f64) {
        let n = self.npoints();
        if n == 0 {
            return (Vec::new(), 0.0, 1.0);
        }

        let y_mean: f64 = self.energies.iter().sum::<f64>() / n as f64;
        let variance: f64 = self.energies.iter().map(|e| (e - y_mean).powi(2)).sum::<f64>()
            / (n as f64 - 1.0).max(1.0);
        let y_std = variance.sqrt().max(1e-10);

        let mut y_full = Vec::with_capacity(n + self.gradients.len());
        for &e in &self.energies {
            y_full.push((e - y_mean) / y_std);
        }
        for &g in &self.gradients {
            y_full.push(g / y_std);
        }

        (y_full, y_mean, y_std)
    }

    /// Extract a subset by indices with pre-allocation for efficiency.
    pub fn extract_subset(&self, indices: &[usize]) -> TrainingData {
        let mut sub = TrainingData::new(self.dim);
        // Pre-allocate to avoid repeated reallocations
        sub.data.reserve(indices.len() * self.dim);
        sub.energies.reserve(indices.len());
        sub.gradients.reserve(indices.len() * self.dim);
        for &i in indices {
            sub.data.extend_from_slice(self.col(i));
            sub.energies.push(self.energies[i]);
            let gs = i * self.dim;
            sub.gradients
                .extend_from_slice(&self.gradients[gs..gs + self.dim]);
        }
        sub
    }
}

/// Gaussian process model with derivative observations.
#[derive(Debug, Clone)]
pub struct GPModel {
    pub kernel: Kernel,
    /// Training inputs stored column-major: D*N flat.
    pub x_data: Vec<f64>,
    pub dim: usize,
    pub n_train: usize,
    /// Training targets [energies; gradients], length N*(1+D).
    pub y: Vec<f64>,
    pub noise_var: f64,
    pub grad_noise_var: f64,
    pub jitter: f64,
    /// Constant kernel variance: k_c(x,x') = const_sigma2 for energy-energy block.
    /// Matches C++ gpr_optim ConstantCF: max(1, mean_y^2). Not optimized by SCG.
    pub const_sigma2: f64,
    /// SCG initial lambda (trust-region regularization). Default 10.0 (C++ default).
    /// d_000 config uses 100. Higher = more conservative SCG steps.
    pub scg_lambda_init: f64,
    /// Student-t prior degrees of freedom on sqrt(sigma2). 0.0 = Gaussian fallback.
    /// C++ gpr_optim uses PriorSqrtt with dof=28 for molecular systems.
    pub prior_dof: f64,
    /// Student-t prior scale parameter (s2). Default 1.0.
    pub prior_s2: f64,
    /// Student-t prior location parameter (mu). Default 0.0.
    pub prior_mu: f64,
}

impl GPModel {
    /// Create a new GP model with validation.
    ///
    /// # Errors
    ///
    /// Returns `Err` if training data is empty, or if any values are non-finite.
    pub fn new(
        kernel: Kernel,
        td: &TrainingData,
        y: Vec<f64>,
        noise_var: f64,
        grad_noise_var: f64,
        jitter: f64,
    ) -> GpResult<Self> {
        // Validate training data
        if td.npoints() == 0 {
            return Err(GpError::EmptyTrainingData);
        }
        
        // Validate y targets
        if y.is_empty() {
            return Err(GpError::EmptyTargets);
        }
        for (i, &val) in y.iter().enumerate() {
            if !val.is_finite() {
                return Err(GpError::NonFiniteTargets { index: i, value: val });
            }
        }
        
        // Validate kernel parameters
        let sigma2 = kernel.signal_variance();
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return Err(GpError::InvalidKernelParams {
                param: "signal_variance".to_string(),
                value: sigma2,
            });
        }
        for (i, &ls) in kernel.inv_lengthscales().iter().enumerate() {
            if ls <= 0.0 || !ls.is_finite() {
                return Err(GpError::InvalidKernelParams {
                    param: format!("inv_lengthscale[{}]", i),
                    value: ls,
                });
            }
        }

        let dim = td.dim;
        let n_train = td.npoints();
        // Constant kernel: off by default (0.0). Set to 1.0 for molecular
        // systems to match C++ gpr_optim ConstantCF. Callers override via
        // gp.const_sigma2 = config.const_sigma2 after construction.
        let const_sigma2 = 0.0;
        Ok(Self {
            kernel,
            x_data: td.data.clone(),
            dim,
            n_train,
            y,
            noise_var,
            grad_noise_var,
            jitter,
            const_sigma2,
            scg_lambda_init: 10.0,
            prior_dof: 0.0,
            prior_s2: 1.0,
            prior_mu: 0.0,
        })
    }

    /// Get training column i.
    pub fn train_col(&self, i: usize) -> &[f64] {
        let start = i * self.dim;
        &self.x_data[start..start + self.dim]
    }
}

/// Inverse normal CDF at p=0.75: norminv(0.75, 0, 1).
pub const NORMINV_075: f64 = 0.6744897501960817;

/// Data-dependent initialization of MolInvDistSE hyperparameters.
/// Matches MATLAB GPstuff initialization.
pub fn init_mol_invdist_se(
    td: &TrainingData,
    kernel: &crate::kernel::MolInvDistSE,
) -> crate::kernel::MolInvDistSE {
    let n = td.npoints();
    let frozen = &kernel.frozen_coords;

    // Compute inverse distance features for all training points
    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| crate::invdist::compute_inverse_distances(td.col(i), frozen))
        .collect();

    // Max pairwise distance in feature space
    let mut max_feat_dist = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = features[i]
                .iter()
                .zip(features[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            max_feat_dist = max_feat_dist.max(d);
        }
    }

    // Energy range
    let range_y = td
        .energies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - td.energies
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
    let range_y = range_y.max(1e-10);

    // sqrt(2) factor from MATLAB dist_at
    let range_x = (2.0f64).sqrt() * max_feat_dist.max(1e-10);

    let sigma2 = (NORMINV_075 * range_y / 3.0).powi(2);
    let ell = NORMINV_075 * range_x / 3.0;
    let inv_ell = 1.0 / ell.max(1e-10);

    let n_ls = kernel.inv_lengthscales.len();
    kernel.with_params(sigma2, vec![inv_ell; n_ls])
}

/// Data-dependent initialization for CartesianSE.
///
/// Same approach as MolInvDistSE but features = coordinates directly.
pub fn init_cartesian_se(
    td: &TrainingData,
    kernel: &crate::kernel::CartesianSE,
) -> crate::kernel::CartesianSE {
    let n = td.npoints();

    // Max pairwise distance in coordinate space
    let mut max_dist = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = td.col(i)
                .iter()
                .zip(td.col(j).iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            max_dist = max_dist.max(d);
        }
    }

    let range_y = td
        .energies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - td.energies
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
    let range_y = range_y.max(1e-10);

    let range_x = (2.0f64).sqrt() * max_dist.max(1e-10);

    let sigma2 = (NORMINV_075 * range_y / 3.0).powi(2);
    let ell = NORMINV_075 * range_x / 3.0;
    let inv_ell = 1.0 / ell.max(1e-10);

    kernel.with_params(sigma2, inv_ell)
}

/// Data-dependent initialization dispatching on Kernel variant.
pub fn init_kernel(td: &TrainingData, kernel: &Kernel) -> Kernel {
    match kernel {
        Kernel::MolInvDist(k) => Kernel::MolInvDist(init_mol_invdist_se(td, k)),
        Kernel::Cartesian(k) => Kernel::Cartesian(init_cartesian_se(td, k)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_data() {
        let mut td = TrainingData::new(3);
        td.add_point(&[1.0, 2.0, 3.0], 0.5, &[0.1, 0.2, 0.3]).expect("test setup: add_point should succeed with valid data");
        td.add_point(&[4.0, 5.0, 6.0], 1.5, &[0.4, 0.5, 0.6]).expect("test setup: add_point should succeed with valid data");
        assert_eq!(td.npoints(), 2);
        assert_eq!(td.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(td.col(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_normalize() {
        let mut td = TrainingData::new(2);
        td.add_point(&[0.0, 0.0], 1.0, &[0.1, 0.2]).expect("test setup: add_point should succeed with valid data");
        td.add_point(&[1.0, 1.0], 3.0, &[0.3, 0.4]).expect("test setup: add_point should succeed with valid data");
        let (y, mean, std) = td.normalize();
        assert!((mean - 2.0).abs() < 1e-10);
        assert!(std > 0.0);
        assert_eq!(y.len(), 6); // 2 energies + 4 gradients
    }

    #[test]
    #[should_panic(expected = "dimension cannot be 0")]
    fn test_validation_zero_dim_panics() {
        // Test empty dimension causes panic
        let _td = TrainingData::new(0);
        
        // Test NaN rejection
        let mut td = TrainingData::new(2);
        assert!(matches!(
            td.add_point(&[f64::NAN, 1.0], 0.5, &[0.1, 0.2]),
            Err(GpError::NonFiniteData { .. })
        ));
        
        // Test dimension mismatch
        assert!(matches!(
            td.add_point(&[1.0], 0.5, &[0.1, 0.2]),
            Err(GpError::DimensionMismatch { .. })
        ));
    }
}
