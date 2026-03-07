//! Error types for GP operations.
//!
//! Provides comprehensive error handling for all GP operations,
//! replacing unsafe `.unwrap()` calls with proper `Result<T, GpError>`.

use std::fmt;

/// Errors that can occur during GP operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GpError {
    /// Training data is empty (no points available).
    EmptyTrainingData,
    /// Training data contains non-finite values (NaN or Inf).
    NonFiniteData { index: usize, value: f64 },
    /// Energy values contain non-finite values.
    NonFiniteEnergy { index: usize, value: f64 },
    /// Gradient values contain non-finite values.
    NonFiniteGradient { index: usize, value: f64 },
    /// Target values contain non-finite values.
    NonFiniteTargets { index: usize, value: f64 },
    /// Dimension mismatch between data and expected dimension.
    DimensionMismatch { expected: usize, actual: usize },
    /// Kernel hyperparameters are invalid (negative variance, zero lengthscale).
    InvalidKernelParams { param: String, value: f64 },
    /// Cholesky decomposition failed (matrix not positive definite).
    CholeskyFailed { attempts: usize },
    /// SCG optimization failed to converge.
    ScgDidNotConverge,
    /// Index out of bounds for training data.
    IndexOutOfBounds { index: usize, max: usize },
    /// No training data available for prediction.
    NoTrainingData,
    /// Energy or gradient values are empty.
    EmptyTargets,
    /// Feature dimension mismatch.
    FeatureDimensionMismatch { expected: usize, actual: usize },
    /// RFF construction failed.
    RffConstructionFailed(String),
    /// Prediction failed.
    PredictionFailed(String),
}

impl fmt::Display for GpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpError::EmptyTrainingData => write!(f, "Training data is empty"),
            GpError::NonFiniteData { index, value } => {
                write!(f, "Non-finite value in training data at index {}: {}", index, value)
            }
            GpError::NonFiniteEnergy { index, value } => {
                write!(f, "Non-finite energy at index {}: {}", index, value)
            }
            GpError::NonFiniteGradient { index, value } => {
                write!(f, "Non-finite gradient at index {}: {}", index, value)
            }
            GpError::NonFiniteTargets { index, value } => {
                write!(f, "Non-finite target at index {}: {}", index, value)
            }
            GpError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            GpError::InvalidKernelParams { param, value } => {
                write!(f, "Invalid kernel parameter '{}': {}", param, value)
            }
            GpError::CholeskyFailed { attempts } => {
                write!(f, "Cholesky decomposition failed after {} attempts", attempts)
            }
            GpError::ScgDidNotConverge => write!(f, "SCG optimization did not converge"),
            GpError::IndexOutOfBounds { index, max } => {
                write!(f, "Index {} out of bounds (max: {})", index, max)
            }
            GpError::NoTrainingData => write!(f, "No training data available"),
            GpError::EmptyTargets => write!(f, "Target values are empty"),
            GpError::FeatureDimensionMismatch { expected, actual } => {
                write!(f, "Feature dimension mismatch: expected {}, got {}", expected, actual)
            }
            GpError::RffConstructionFailed(msg) => write!(f, "RFF construction failed: {}", msg),
            GpError::PredictionFailed(msg) => write!(f, "Prediction failed: {}", msg),
        }
    }
}

impl std::error::Error for GpError {}

/// Result type alias for GP operations.
pub type GpResult<T> = Result<T, GpError>;

/// Validation helpers for GP data.
pub mod validate {
    use super::{GpError, GpResult};

    /// Check if a value is finite (not NaN or Inf).
    pub fn is_finite(x: f64) -> bool {
        x.is_finite()
    }

    /// Validate that all values in a slice are finite.
    /// Returns the index and value of the first non-finite entry.
    pub fn validate_finite_slice(data: &[f64]) -> GpResult<()> {
        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(GpError::NonFiniteData { index: i, value: val });
            }
        }
        Ok(())
    }

    /// Validate training data dimension.
    pub fn validate_dimension(expected: usize, actual: usize) -> GpResult<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(GpError::DimensionMismatch { expected, actual })
        }
    }

    /// Validate kernel signal variance (must be positive).
    pub fn validate_signal_variance(sigma2: f64) -> GpResult<()> {
        if sigma2 > 0.0 && sigma2.is_finite() {
            Ok(())
        } else {
            Err(GpError::InvalidKernelParams {
                param: "signal_variance".to_string(),
                value: sigma2,
            })
        }
    }

    /// Validate kernel lengthscale (must be positive and finite).
    pub fn validate_lengthscale(ell: f64) -> GpResult<()> {
        if ell > 0.0 && ell.is_finite() {
            Ok(())
        } else {
            Err(GpError::InvalidKernelParams {
                param: "lengthscale".to_string(),
                value: ell,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_finite_slice() {
        let good_data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(validate::validate_finite_slice(&good_data).is_ok());

        let bad_nan = vec![1.0, f64::NAN, 3.0];
        assert!(matches!(
            validate::validate_finite_slice(&bad_nan),
            Err(GpError::NonFiniteData { .. })
        ));

        let bad_inf = vec![1.0, f64::INFINITY, 3.0];
        assert!(matches!(
            validate::validate_finite_slice(&bad_inf),
            Err(GpError::NonFiniteData { .. })
        ));
    }

    #[test]
    fn test_validate_dimension() {
        assert!(validate::validate_dimension(3, 3).is_ok());
        assert!(matches!(
            validate::validate_dimension(3, 4),
            Err(GpError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_kernel_params() {
        assert!(validate::validate_signal_variance(1.0).is_ok());
        assert!(matches!(
            validate::validate_signal_variance(-1.0),
            Err(GpError::InvalidKernelParams { .. })
        ));
        assert!(matches!(
            validate::validate_signal_variance(f64::NAN),
            Err(GpError::InvalidKernelParams { .. })
        ));

        assert!(validate::validate_lengthscale(0.5).is_ok());
        assert!(matches!(
            validate::validate_lengthscale(0.0),
            Err(GpError::InvalidKernelParams { .. })
        ));
    }

    #[test]
    fn test_error_display() {
        let err = GpError::EmptyTrainingData;
        assert_eq!(format!("{}", err), "Training data is empty");

        let err = GpError::NonFiniteData { index: 5, value: f64::NAN };
        assert!(format!("{}", err).contains("index 5"));
    }
}
