//! Shared helpers for benchmark-oriented examples and workflow runners.

use crate::prior_mean::{
    select_best_candidate_by_gradient_match, PriorCandidate, PriorMeanConfig,
};
use crate::types::TrainingData;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkVariant {
    Chemgp,
    PhysicalPrior,
    AdaptivePrior,
    RecycledLocalPes,
}

impl BenchmarkVariant {
    pub fn from_env() -> Self {
        let raw = std::env::var("CHEMGP_BENCH_VARIANT")
            .unwrap_or_else(|_| "chemgp".to_string())
            .to_ascii_lowercase();
        match raw.as_str() {
            "chemgp" | "baseline" => Self::Chemgp,
            "physical_prior" | "physical-prior" => Self::PhysicalPrior,
            "adaptive_prior" | "adaptive-prior" => Self::AdaptivePrior,
            "recycled_local_pes" | "recycled-local-pes" | "meta_gp" | "meta-gp" => {
                Self::RecycledLocalPes
            }
            other => panic!(
                "Unknown CHEMGP_BENCH_VARIANT '{}'; expected chemgp, physical_prior, adaptive_prior, or recycled_local_pes",
                other
            ),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Chemgp => "chemgp",
            Self::PhysicalPrior => "physical_prior",
            Self::AdaptivePrior => "adaptive_prior",
            Self::RecycledLocalPes => "recycled_local_pes",
        }
    }

    pub fn uses_prior(self) -> bool {
        !matches!(self, Self::Chemgp)
    }
}

pub fn output_path(default_name: &str) -> String {
    std::env::var("CHEMGP_BENCH_OUTPUT").unwrap_or_else(|_| default_name.to_string())
}

pub fn artifact_path(default_name: &str) -> String {
    match std::env::var("CHEMGP_BENCH_ARTIFACT_DIR") {
        Ok(dir) => std::path::Path::new(&dir)
            .join(default_name)
            .display()
            .to_string(),
        Err(_) => default_name.to_string(),
    }
}

pub fn linear_prior(center: &[f64], energy: f64, gradient: &[f64], label: &str) -> PriorMeanConfig {
    PriorMeanConfig::from_candidate(&PriorCandidate::linear(
        label,
        center.to_vec(),
        energy,
        gradient.to_vec(),
    ))
}

pub fn nearest_linear_prior(observations: &[(&str, &[f64], f64, &[f64])]) -> PriorMeanConfig {
    let candidates = observations
        .iter()
        .map(|(label, center, energy, gradient)| {
            PriorCandidate::linear(*label, center.to_vec(), *energy, gradient.to_vec())
        })
        .collect();
    PriorMeanConfig::NearestTaylor { candidates }
}

pub fn sampled_taylor_prior(
    oracle: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    points: &[Vec<f64>],
    label_prefix: &str,
) -> PriorMeanConfig {
    let candidates = points
        .iter()
        .enumerate()
        .map(|(idx, x)| {
            let (energy, gradient) = oracle(x);
            PriorCandidate::linear(
                format!("{label_prefix}_{idx}"),
                x.clone(),
                energy,
                gradient,
            )
        })
        .collect();
    PriorMeanConfig::NearestTaylor { candidates }
}

pub fn select_adaptive_prior(
    x: &[f64],
    energy: f64,
    gradient: &[f64],
    observations: &[(&str, &[f64], f64, &[f64])],
) -> PriorMeanConfig {
    select_adaptive_prior_with_label(x, energy, gradient, observations).0
}

pub fn select_adaptive_prior_with_label(
    x: &[f64],
    energy: f64,
    gradient: &[f64],
    observations: &[(&str, &[f64], f64, &[f64])],
) -> (PriorMeanConfig, String) {
    let candidates: Vec<PriorCandidate> = observations
        .iter()
        .map(|(label, center, energy, gradient)| {
            PriorCandidate::linear(*label, center.to_vec(), *energy, gradient.to_vec())
        })
        .collect();
    let best_idx = select_best_candidate_by_gradient_match(x, energy, gradient, &candidates);
    (
        PriorMeanConfig::from_candidate(&candidates[best_idx]),
        candidates[best_idx].label.clone(),
    )
}

pub fn nearest_prior_library_label(labels: &[&str]) -> String {
    format!("nearest:[{}]", labels.join(","))
}

#[cfg(test)]
mod tests {
    use super::{nearest_prior_library_label, select_adaptive_prior_with_label, BenchmarkVariant};
    use crate::prior_mean::PriorMeanConfig;

    #[test]
    fn benchmark_variant_env_aliases_work() {
        std::env::set_var("CHEMGP_BENCH_VARIANT", "meta_gp");
        assert_eq!(BenchmarkVariant::from_env(), BenchmarkVariant::RecycledLocalPes);
        std::env::set_var("CHEMGP_BENCH_VARIANT", "physical-prior");
        assert_eq!(BenchmarkVariant::from_env(), BenchmarkVariant::PhysicalPrior);
        std::env::remove_var("CHEMGP_BENCH_VARIANT");
    }

    #[test]
    fn adaptive_prior_returns_selected_label() {
        let (cfg, label) = select_adaptive_prior_with_label(
            &[0.0],
            1.0,
            &[2.0],
            &[
                ("good", &[0.0], 1.0, &[2.0]),
                ("bad", &[0.0], 0.0, &[0.0]),
            ],
        );
        assert_eq!(label, "good");
        match cfg {
            PriorMeanConfig::TaylorDiagonal { energy_offset, .. } => {
                assert!((energy_offset - 1.0).abs() < 1e-12);
            }
            _ => panic!("expected TaylorDiagonal prior"),
        }
    }

    #[test]
    fn nearest_library_label_is_stable() {
        assert_eq!(
            nearest_prior_library_label(&["reactant", "product"]),
            "nearest:[reactant,product]"
        );
    }
}

pub fn seed_training_data(
    oracle: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    x_init: &[f64],
    n_initial_perturb: usize,
    perturb_scale: f64,
    seed: u64,
) -> (TrainingData, Vec<(Vec<f64>, f64, Vec<f64>)>) {
    let mut td = TrainingData::new(x_init.len());
    let mut observations = Vec::with_capacity(n_initial_perturb + 1);

    let (e0, g0) = oracle(x_init);
    td.add_point(x_init, e0, &g0)
        .expect("Failed to seed benchmark training data at initial point");
    observations.push((x_init.to_vec(), e0, g0));

    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n_initial_perturb {
        let perturb: Vec<f64> = (0..x_init.len())
            .map(|_| (rng.random::<f64>() - 0.5) * perturb_scale)
            .collect();
        let x_p: Vec<f64> = x_init
            .iter()
            .zip(perturb.iter())
            .map(|(a, b)| a + b)
            .collect();
        let (e_p, g_p) = oracle(&x_p);
        td.add_point(&x_p, e_p, &g_p)
            .expect("Failed to add benchmark perturbation point");
        observations.push((x_p, e_p, g_p));
    }

    (td, observations)
}
