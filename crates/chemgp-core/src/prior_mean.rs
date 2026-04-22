//! Prior-mean specifications for residualized GP training and prediction.

use crate::types::TrainingData;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PriorMeanConfig {
    /// Zero energy and zero gradient prior.
    Zero,
    /// Use the first observed training energy as a constant prior offset.
    /// This preserves the current ChemGP centering behavior by default.
    Reference,
    /// User-specified constant energy prior with zero gradients.
    Constant { energy: f64 },
    /// Diagonal quadratic prior:
    /// E(x) = e0 + 0.5 * sum_i k_i * (x_i - c_i)^2
    /// grad_i = k_i * (x_i - c_i)
    Quadratic {
        center: Vec<f64>,
        energy_offset: f64,
        curvature: Vec<f64>,
    },
}

impl Default for PriorMeanConfig {
    fn default() -> Self {
        Self::Reference
    }
}

impl PriorMeanConfig {
    pub fn evaluate(&self, x: &[f64], reference_energy: f64) -> (f64, Vec<f64>) {
        match self {
            Self::Zero => (0.0, vec![0.0; x.len()]),
            Self::Reference => (reference_energy, vec![0.0; x.len()]),
            Self::Constant { energy } => (*energy, vec![0.0; x.len()]),
            Self::Quadratic { center, energy_offset, curvature } => {
                assert_eq!(
                    center.len(),
                    x.len(),
                    "PriorMeanConfig::Quadratic center length must match x length",
                );
                assert_eq!(
                    curvature.len(),
                    x.len(),
                    "PriorMeanConfig::Quadratic curvature length must match x length",
                );
                let mut e = *energy_offset;
                let mut g = vec![0.0; x.len()];
                for i in 0..x.len() {
                    let dx = x[i] - center[i];
                    e += 0.5 * curvature[i] * dx * dx;
                    g[i] = curvature[i] * dx;
                }
                (e, g)
            }
        }
    }

    pub fn residualize_training_data(&self, td: &TrainingData) -> (Vec<f64>, Vec<f64>) {
        let reference_energy = td.energies.first().copied().unwrap_or(0.0);
        let mut residual_energies = Vec::with_capacity(td.npoints());
        let mut residual_gradients = Vec::with_capacity(td.gradients.len());

        for i in 0..td.npoints() {
            let x = td.col(i);
            let (prior_e, prior_g) = self.evaluate(x, reference_energy);
            residual_energies.push(td.energies[i] - prior_e);

            let g_start = i * td.dim;
            for (j, prior_gj) in prior_g.iter().enumerate().take(td.dim) {
                residual_gradients.push(td.gradients[g_start + j] - prior_gj);
            }
        }

        (residual_energies, residual_gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::PriorMeanConfig;
    use crate::types::TrainingData;

    #[test]
    fn reference_prior_uses_first_training_energy() {
        let mut td = TrainingData::new(2);
        td.add_point(&[0.0, 0.0], 3.5, &[0.0, 0.0]).unwrap();
        td.add_point(&[1.0, 0.0], 4.5, &[1.0, 0.0]).unwrap();

        let (residual_e, residual_g) = PriorMeanConfig::Reference.residualize_training_data(&td);
        assert_eq!(residual_e, vec![0.0, 1.0]);
        assert_eq!(residual_g, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn constant_prior_subtracts_energy_but_not_gradients() {
        let mut td = TrainingData::new(2);
        td.add_point(&[0.0, 0.0], 5.0, &[1.0, -1.0]).unwrap();

        let (residual_e, residual_g) =
            PriorMeanConfig::Constant { energy: 2.0 }.residualize_training_data(&td);
        assert_eq!(residual_e, vec![3.0]);
        assert_eq!(residual_g, vec![1.0, -1.0]);
    }

    #[test]
    fn quadratic_prior_returns_energy_and_gradient() {
        let prior = PriorMeanConfig::Quadratic {
            center: vec![1.0, -1.0],
            energy_offset: 2.0,
            curvature: vec![4.0, 2.0],
        };

        let (e, g) = prior.evaluate(&[2.0, 1.0], 0.0);
        assert!((e - 8.0).abs() < 1e-12);
        assert_eq!(g, vec![4.0, 4.0]);
    }
}
