pub mod invdist;
pub mod kernel;
pub mod types;
pub mod covariance;
pub mod predict;
pub mod nll;
pub mod scg;
pub mod train;
pub mod distances;
pub mod emd;
pub mod sampling;
pub mod lbfgs;
pub mod optim_step;
pub mod trust;
pub mod minimize;
pub mod rff;
pub mod neb_path;
pub mod idpp;
pub mod neb;
pub mod neb_oie;
pub mod dimer;
pub mod dimer_utils;
pub mod hod;
pub mod otgpd;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "rgpot")]
pub mod oracle;

/// Analytical potentials: LJ, Muller-Brown, LEPS.
pub mod potentials;

/// Why an optimizer terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StopReason {
    Converged,
    MaxIterations,
    OracleCap,
    ForceStagnation,
    UserCallback,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Converged => write!(f, "CONVERGED"),
            StopReason::MaxIterations => write!(f, "MAX_ITERATIONS"),
            StopReason::OracleCap => write!(f, "ORACLE_CAP"),
            StopReason::ForceStagnation => write!(f, "FORCE_STAGNATION"),
            StopReason::UserCallback => write!(f, "USER_CALLBACK"),
        }
    }
}
