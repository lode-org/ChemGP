//! Prior-mean specifications for residualized GP training and prediction.

use crate::types::TrainingData;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PriorCandidate {
    pub label: String,
    pub center: Vec<f64>,
    pub energy_offset: f64,
    pub gradient: Vec<f64>,
    pub curvature: Vec<f64>,
}

impl PriorCandidate {
    pub fn new(
        label: impl Into<String>,
        center: Vec<f64>,
        energy_offset: f64,
        gradient: Vec<f64>,
        curvature: Vec<f64>,
    ) -> Self {
        assert_eq!(
            center.len(),
            gradient.len(),
            "PriorCandidate gradient length must match center length",
        );
        assert_eq!(
            center.len(),
            curvature.len(),
            "PriorCandidate curvature length must match center length",
        );
        Self {
            label: label.into(),
            center,
            energy_offset,
            gradient,
            curvature,
        }
    }

    pub fn linear(
        label: impl Into<String>,
        center: Vec<f64>,
        energy_offset: f64,
        gradient: Vec<f64>,
    ) -> Self {
        let curvature = vec![0.0; center.len()];
        Self::new(label, center, energy_offset, gradient, curvature)
    }

    pub fn evaluate(&self, x: &[f64]) -> (f64, Vec<f64>) {
        assert_eq!(
            self.center.len(),
            x.len(),
            "PriorCandidate center length must match x length",
        );
        let mut e = self.energy_offset;
        let mut g = self.gradient.clone();
        for i in 0..x.len() {
            let dx = x[i] - self.center[i];
            e += self.gradient[i] * dx + 0.5 * self.curvature[i] * dx * dx;
            g[i] += self.curvature[i] * dx;
        }
        (e, g)
    }

    pub fn distance2(&self, x: &[f64]) -> f64 {
        assert_eq!(
            self.center.len(),
            x.len(),
            "PriorCandidate center length must match x length",
        );
        self.center
            .iter()
            .zip(x.iter())
            .map(|(a, b)| {
                let dx = a - b;
                dx * dx
            })
            .sum()
    }
}

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
    /// Local first/second-order Taylor prior around a reference structure:
    /// E(x) = e0 + g0·(x-c) + 0.5 * sum_i k_i * (x_i - c_i)^2
    /// grad_i = g0_i + k_i * (x_i - c_i)
    TaylorDiagonal {
        center: Vec<f64>,
        energy_offset: f64,
        gradient: Vec<f64>,
        curvature: Vec<f64>,
    },
    /// Choose the nearest local PES prior from a candidate library.
    NearestTaylor {
        candidates: Vec<PriorCandidate>,
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
            Self::TaylorDiagonal {
                center,
                energy_offset,
                gradient,
                curvature,
            } => {
                let candidate = PriorCandidate::new(
                    "taylor-diagonal",
                    center.clone(),
                    *energy_offset,
                    gradient.clone(),
                    curvature.clone(),
                );
                candidate.evaluate(x)
            }
            Self::NearestTaylor { candidates } => {
                assert!(
                    !candidates.is_empty(),
                    "PriorMeanConfig::NearestTaylor requires at least one candidate",
                );
                let best = candidates
                    .iter()
                    .min_by(|a, b| {
                        a.distance2(x)
                            .partial_cmp(&b.distance2(x))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("NearestTaylor candidate selection failed");
                best.evaluate(x)
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

    pub fn from_candidate(candidate: &PriorCandidate) -> Self {
        Self::TaylorDiagonal {
            center: candidate.center.clone(),
            energy_offset: candidate.energy_offset,
            gradient: candidate.gradient.clone(),
            curvature: candidate.curvature.clone(),
        }
    }
}

pub fn candidate_residual_score(td: &TrainingData, candidate: &PriorCandidate) -> f64 {
    let mut score = 0.0;
    for i in 0..td.npoints() {
        let x = td.col(i);
        let (prior_e, prior_g) = candidate.evaluate(x);
        let e_res = td.energies[i] - prior_e;
        score += e_res * e_res;
        let g_start = i * td.dim;
        for (j, prior_gj) in prior_g.iter().enumerate().take(td.dim) {
            let g_res = td.gradients[g_start + j] - prior_gj;
            score += g_res * g_res;
        }
    }
    score
}

pub fn select_best_candidate(td: &TrainingData, candidates: &[PriorCandidate]) -> usize {
    assert!(
        !candidates.is_empty(),
        "select_best_candidate requires at least one candidate",
    );
    candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            candidate_residual_score(td, a)
                .partial_cmp(&candidate_residual_score(td, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .expect("candidate selection failed")
}

#[cfg(test)]
mod tests {
    use super::{candidate_residual_score, select_best_candidate, PriorCandidate, PriorMeanConfig};
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

    #[test]
    fn taylor_diagonal_prior_returns_energy_and_gradient() {
        let prior = PriorMeanConfig::TaylorDiagonal {
            center: vec![0.5, -0.5],
            energy_offset: 1.0,
            gradient: vec![2.0, -1.0],
            curvature: vec![4.0, 2.0],
        };

        let (e, g) = prior.evaluate(&[1.0, 1.0], 0.0);
        assert!((e - 3.25).abs() < 1e-12);
        assert_eq!(g, vec![4.0, 2.0]);
    }

    #[test]
    fn nearest_taylor_prior_selects_closest_candidate() {
        let prior = PriorMeanConfig::NearestTaylor {
            candidates: vec![
                PriorCandidate::linear("left", vec![0.0, 0.0], 1.0, vec![1.0, 0.0]),
                PriorCandidate::linear("right", vec![10.0, 0.0], 3.0, vec![0.0, 2.0]),
            ],
        };

        let (e, g) = prior.evaluate(&[9.0, 0.0], 0.0);
        assert!((e - 3.0).abs() < 1e-12);
        assert_eq!(g, vec![0.0, 2.0]);
    }

    #[test]
    fn candidate_selection_prefers_lower_residual_prior() {
        let mut td = TrainingData::new(1);
        td.add_point(&[0.0], 1.0, &[2.0]).unwrap();
        td.add_point(&[1.0], 3.0, &[2.0]).unwrap();

        let good = PriorCandidate::linear("good", vec![0.0], 1.0, vec![2.0]);
        let bad = PriorCandidate::linear("bad", vec![0.0], 0.0, vec![0.0]);

        assert!(candidate_residual_score(&td, &good) < candidate_residual_score(&td, &bad));
        assert_eq!(select_best_candidate(&td, &[bad, good]), 1);
    }
}
