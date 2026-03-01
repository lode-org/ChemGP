//! GP kernels: MolInvDistSE (molecular inverse-distance) and CartesianSE (raw coordinates).
//!
//! The `Kernel` enum dispatches between kernel types throughout the GP pipeline.
//!
//! MolInvDistSE: k(x,y) = sigma^2 * exp(-sum_i (theta_i * (f_i(x) - f_i(y)))^2)
//!   where f_i are inverse interatomic distances (1/r_ij).
//!
//! CartesianSE: k(x,y) = sigma^2 * exp(-theta^2 * ||x - y||^2)
//!   operates directly on raw coordinates (2D, 3D, or arbitrary dimension).

use crate::invdist::{build_feature_map, compute_inverse_distances, PairScheme};
use faer::Mat;

/// Squared Exponential kernel on inverse interatomic distance features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MolInvDistSE {
    pub signal_variance: f64,
    pub inv_lengthscales: Vec<f64>,
    pub frozen_coords: Vec<f64>,
    /// Maps each feature index to its lengthscale parameter index (empty = isotropic).
    pub feature_params_map: Vec<usize>,
}

/// Squared Exponential kernel on raw Cartesian coordinates.
///
/// For non-molecular surfaces (Muller-Brown, LEPS-2D, etc.) where features = coordinates.
/// Isotropic: single inv_lengthscale for all dimensions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CartesianSE {
    pub signal_variance: f64,
    pub inv_lengthscale: f64,
}

/// Kernel enum: dispatches between MolInvDistSE and CartesianSE.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Kernel {
    MolInvDist(MolInvDistSE),
    Cartesian(CartesianSE),
}

/// Kernel block output: (k_ee, k_ef, k_fe, k_ff).
pub struct KernelBlocks {
    pub k_ee: f64,
    pub k_ef: Vec<f64>,  // length D (energy-force cross-covariance row)
    pub k_fe: Vec<f64>,  // length D (force-energy cross-covariance col)
    pub k_ff: Mat<f64>,  // D x D force-force block
}

/// Kernel blocks + hyperparameter gradients.
pub struct KernelBlocksWithGrads {
    pub blocks: KernelBlocks,
    /// One KernelBlocks per log-space parameter [log(sigma2), log(theta_1), ...].
    pub grad_blocks: Vec<KernelBlocks>,
}

// ---------------------------------------------------------------------------
// MolInvDistSE implementation (unchanged)
// ---------------------------------------------------------------------------

impl MolInvDistSE {
    /// Isotropic constructor (single lengthscale for all pairs).
    pub fn isotropic(signal_variance: f64, inv_ls: f64, frozen: Vec<f64>) -> Self {
        Self {
            signal_variance,
            inv_lengthscales: vec![inv_ls],
            frozen_coords: frozen,
            feature_params_map: Vec::new(),
        }
    }

    /// Type-aware constructor from a PairScheme.
    pub fn from_pair_scheme(
        signal_variance: f64,
        inv_ls: f64,
        frozen: Vec<f64>,
        scheme: &PairScheme,
    ) -> Self {
        let n_mov = scheme.mov_types.len();
        let n_fro = scheme.fro_types.len();
        let feat_map = build_feature_map(
            n_mov, n_fro, &scheme.mov_types, &scheme.fro_types, &scheme.pair_map,
        );
        Self {
            signal_variance,
            inv_lengthscales: vec![inv_ls; scheme.n_params],
            frozen_coords: frozen,
            feature_params_map: feat_map,
        }
    }

    /// Atomic-number constructor: builds pair-type scheme automatically.
    pub fn from_atomic_numbers(
        atomic_numbers_mov: &[i32],
        frozen_coords: Vec<f64>,
        atomic_numbers_fro: &[i32],
        signal_variance: f64,
        inv_lengthscale: f64,
    ) -> Self {
        let scheme = crate::invdist::build_pair_scheme(atomic_numbers_mov, atomic_numbers_fro);
        Self::from_pair_scheme(signal_variance, inv_lengthscale, frozen_coords, &scheme)
    }

    /// Reconstruct kernel with new hyperparameters, preserving structure.
    pub fn with_params(&self, signal_variance: f64, inv_lengthscales: Vec<f64>) -> Self {
        Self {
            signal_variance,
            inv_lengthscales,
            frozen_coords: self.frozen_coords.clone(),
            feature_params_map: self.feature_params_map.clone(),
        }
    }

    /// Evaluate kernel value k(x, y).
    pub fn eval(&self, x: &[f64], y: &[f64]) -> f64 {
        let fx = compute_inverse_distances(x, &self.frozen_coords);
        let fy = compute_inverse_distances(y, &self.frozen_coords);

        let mut d2 = 0.0;
        if !self.feature_params_map.is_empty() {
            for i in 0..fx.len() {
                let idx = self.feature_params_map[i];
                let val = (fx[i] - fy[i]) * self.inv_lengthscales[idx];
                d2 += val * val;
            }
        } else if self.inv_lengthscales.len() == 1 {
            let theta = self.inv_lengthscales[0];
            for i in 0..fx.len() {
                let diff = fx[i] - fy[i];
                d2 += diff * diff;
            }
            d2 *= theta * theta;
        }

        self.signal_variance * (-d2).exp()
    }
}

// ---------------------------------------------------------------------------
// CartesianSE implementation
// ---------------------------------------------------------------------------

impl CartesianSE {
    pub fn new(signal_variance: f64, inv_lengthscale: f64) -> Self {
        Self { signal_variance, inv_lengthscale }
    }

    pub fn with_params(&self, signal_variance: f64, inv_lengthscale: f64) -> Self {
        Self { signal_variance, inv_lengthscale }
    }

    pub fn eval(&self, x: &[f64], y: &[f64]) -> f64 {
        let theta2 = self.inv_lengthscale * self.inv_lengthscale;
        let mut d2 = 0.0;
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            d2 += diff * diff;
        }
        self.signal_variance * (-theta2 * d2).exp()
    }
}

// ---------------------------------------------------------------------------
// Kernel enum dispatch
// ---------------------------------------------------------------------------

impl Kernel {
    pub fn signal_variance(&self) -> f64 {
        match self {
            Kernel::MolInvDist(k) => k.signal_variance,
            Kernel::Cartesian(k) => k.signal_variance,
        }
    }

    pub fn inv_lengthscales(&self) -> Vec<f64> {
        match self {
            Kernel::MolInvDist(k) => k.inv_lengthscales.clone(),
            Kernel::Cartesian(k) => vec![k.inv_lengthscale],
        }
    }

    pub fn frozen_coords(&self) -> &[f64] {
        match self {
            Kernel::MolInvDist(k) => &k.frozen_coords,
            Kernel::Cartesian(_) => &[],
        }
    }

    pub fn feature_params_map(&self) -> &[usize] {
        match self {
            Kernel::MolInvDist(k) => &k.feature_params_map,
            Kernel::Cartesian(_) => &[],
        }
    }

    pub fn n_ls_params(&self) -> usize {
        match self {
            Kernel::MolInvDist(k) => k.inv_lengthscales.len(),
            Kernel::Cartesian(_) => 1,
        }
    }

    /// Reconstruct kernel with new log-space-derived hyperparameters.
    pub fn with_params(&self, signal_variance: f64, inv_lengthscales: Vec<f64>) -> Kernel {
        match self {
            Kernel::MolInvDist(k) => Kernel::MolInvDist(k.with_params(signal_variance, inv_lengthscales)),
            Kernel::Cartesian(_) => Kernel::Cartesian(CartesianSE {
                signal_variance,
                inv_lengthscale: inv_lengthscales[0],
            }),
        }
    }

    pub fn eval(&self, x: &[f64], y: &[f64]) -> f64 {
        match self {
            Kernel::MolInvDist(k) => k.eval(x, y),
            Kernel::Cartesian(k) => k.eval(x, y),
        }
    }

    pub fn kernel_blocks(&self, x1: &[f64], x2: &[f64]) -> KernelBlocks {
        match self {
            Kernel::MolInvDist(k) => molinvdist_kernel_blocks(k, x1, x2),
            Kernel::Cartesian(k) => cartesian_kernel_blocks(k, x1, x2),
        }
    }

    pub fn kernel_blocks_and_hypergrads(&self, x1: &[f64], x2: &[f64]) -> KernelBlocksWithGrads {
        match self {
            Kernel::MolInvDist(k) => molinvdist_kernel_blocks_and_hypergrads(k, x1, x2),
            Kernel::Cartesian(k) => cartesian_kernel_blocks_and_hypergrads(k, x1, x2),
        }
    }

    /// Number of features for this kernel given coordinate dimension.
    pub fn n_features(&self, dim: usize) -> usize {
        match self {
            Kernel::MolInvDist(k) => {
                let n_mov = dim / 3;
                let n_fro = k.frozen_coords.len() / 3;
                n_mov * (n_mov - 1) / 2 + n_mov * n_fro
            }
            Kernel::Cartesian(_) => dim,
        }
    }
}

impl From<MolInvDistSE> for Kernel {
    fn from(k: MolInvDistSE) -> Self {
        Kernel::MolInvDist(k)
    }
}

impl From<CartesianSE> for Kernel {
    fn from(k: CartesianSE) -> Self {
        Kernel::Cartesian(k)
    }
}

// ---------------------------------------------------------------------------
// MolInvDistSE kernel blocks (existing code, renamed)
// ---------------------------------------------------------------------------

/// Compute inverse distance features AND their analytical Jacobian w.r.t. x_flat.
///
/// Returns (features, J) where J is (n_features x D) as faer::Mat.
pub fn invdist_jacobian(x_flat: &[f64], frozen_flat: &[f64]) -> (Vec<f64>, Mat<f64>) {
    let n_mov = x_flat.len() / 3;
    let n_fro = frozen_flat.len() / 3;
    let d = x_flat.len();
    let n_mm = n_mov * (n_mov - 1) / 2;
    let n_mf = n_mov * n_fro;
    let nf = n_mm + n_mf;

    let mut features = vec![0.0; nf];
    let mut j_mat = Mat::<f64>::zeros(nf, d);

    let mut idx = 0;

    // Moving-Moving pairs (upper triangle: j < i)
    for j in 0..n_mov {
        let xj = [x_flat[3 * j], x_flat[3 * j + 1], x_flat[3 * j + 2]];
        for i in (j + 1)..n_mov {
            let xi = [x_flat[3 * i], x_flat[3 * i + 1], x_flat[3 * i + 2]];

            let dx = xi[0] - xj[0];
            let dy = xi[1] - xj[1];
            let dz = xi[2] - xj[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let invr = 1.0 / r;
            let invr3 = invr * invr * invr;

            features[idx] = invr;

            let xi_base = 3 * i;
            let xj_base = 3 * j;
            j_mat[(idx, xi_base)] = -dx * invr3;
            j_mat[(idx, xi_base + 1)] = -dy * invr3;
            j_mat[(idx, xi_base + 2)] = -dz * invr3;
            j_mat[(idx, xj_base)] = dx * invr3;
            j_mat[(idx, xj_base + 1)] = dy * invr3;
            j_mat[(idx, xj_base + 2)] = dz * invr3;

            idx += 1;
        }
    }

    // Moving-Frozen pairs
    if n_fro > 0 {
        for j in 0..n_mov {
            let xj_base = 3 * j;
            let xj = [x_flat[xj_base], x_flat[xj_base + 1], x_flat[xj_base + 2]];
            for fk in 0..n_fro {
                let xf = [
                    frozen_flat[3 * fk],
                    frozen_flat[3 * fk + 1],
                    frozen_flat[3 * fk + 2],
                ];

                let dx = xj[0] - xf[0];
                let dy = xj[1] - xf[1];
                let dz = xj[2] - xf[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let r = r2.sqrt();
                let invr = 1.0 / r;
                let invr3 = invr * invr * invr;

                features[idx] = invr;

                j_mat[(idx, xj_base)] = -dx * invr3;
                j_mat[(idx, xj_base + 1)] = -dy * invr3;
                j_mat[(idx, xj_base + 2)] = -dz * invr3;

                idx += 1;
            }
        }
    }

    (features, j_mat)
}

/// Helper: multiply J^T (D x nf) * v (nf) -> D-vec
fn mat_t_vec(j: &Mat<f64>, v: &[f64]) -> Vec<f64> {
    let nf = j.nrows();
    let d = j.ncols();
    let mut result = vec![0.0; d];
    for col in 0..d {
        let mut sum = 0.0;
        for row in 0..nf {
            sum += j[(row, col)] * v[row];
        }
        result[col] = sum;
    }
    result
}

/// Helper: J1^T (D x nf) * H (nf x nf) * J2 (nf x D) -> Mat D x D
fn jt_h_j(j1: &Mat<f64>, h: &Mat<f64>, j2: &Mat<f64>) -> Mat<f64> {
    let nf = j1.nrows();
    let d = j1.ncols();
    // tmp = H * J2 (nf x D)
    let mut tmp = Mat::<f64>::zeros(nf, d);
    for i in 0..nf {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..nf {
                s += h[(i, k)] * j2[(k, j)];
            }
            tmp[(i, j)] = s;
        }
    }
    // result = J1^T * tmp (D x D)
    let mut result = Mat::<f64>::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..nf {
                s += j1[(k, i)] * tmp[(k, j)];
            }
            result[(i, j)] = s;
        }
    }
    result
}

/// Compute kernel blocks analytically for MolInvDistSE.
pub fn molinvdist_kernel_blocks(k: &MolInvDistSE, x1: &[f64], x2: &[f64]) -> KernelBlocks {
    let frozen = &k.frozen_coords;
    let (f1, j1) = invdist_jacobian(x1, frozen);
    let (f2, j2) = invdist_jacobian(x2, frozen);
    let nf = f1.len();

    // Per-feature theta^2
    let theta2: Vec<f64> = if !k.feature_params_map.is_empty() {
        (0..nf).map(|i| k.inv_lengthscales[k.feature_params_map[i]].powi(2)).collect()
    } else {
        vec![k.inv_lengthscales[0].powi(2); nf]
    };

    let r: Vec<f64> = (0..nf).map(|i| f1[i] - f2[i]).collect();
    let mut d2 = 0.0;
    for i in 0..nf {
        d2 += theta2[i] * r[i] * r[i];
    }
    let kval = k.signal_variance * (-d2).exp();

    let k_ee = kval;

    let mut dk_df2 = vec![0.0; nf];
    let mut dk_df1 = vec![0.0; nf];
    for i in 0..nf {
        let v = 2.0 * kval * theta2[i] * r[i];
        dk_df2[i] = v;
        dk_df1[i] = -v;
    }

    let u: Vec<f64> = (0..nf).map(|i| theta2[i] * r[i]).collect();
    let mut h_feat = Mat::<f64>::zeros(nf, nf);
    for i in 0..nf {
        h_feat[(i, i)] = 2.0 * kval * (theta2[i] - 2.0 * u[i] * u[i]);
        for j in (i + 1)..nf {
            let val = -4.0 * kval * u[i] * u[j];
            h_feat[(i, j)] = val;
            h_feat[(j, i)] = val;
        }
    }

    let k_ef = mat_t_vec(&j2, &dk_df2);
    let k_fe = mat_t_vec(&j1, &dk_df1);
    let k_ff = jt_h_j(&j1, &h_feat, &j2);

    KernelBlocks { k_ee, k_ef, k_fe, k_ff }
}

/// Backward-compatible alias.
pub fn kernel_blocks(k: &MolInvDistSE, x1: &[f64], x2: &[f64]) -> KernelBlocks {
    molinvdist_kernel_blocks(k, x1, x2)
}

/// Compute kernel blocks AND hyperparameter gradients for MolInvDistSE.
pub fn molinvdist_kernel_blocks_and_hypergrads(
    k: &MolInvDistSE,
    x1: &[f64],
    x2: &[f64],
) -> KernelBlocksWithGrads {
    let frozen = &k.frozen_coords;
    let (f1, j1) = invdist_jacobian(x1, frozen);
    let (f2, j2) = invdist_jacobian(x2, frozen);
    let nf = f1.len();
    let n_ls = k.inv_lengthscales.len();
    let n_params = 1 + n_ls;
    let has_map = !k.feature_params_map.is_empty();

    let mut theta2 = vec![0.0; nf];
    let mut fmap = vec![0usize; nf];
    if has_map {
        for i in 0..nf {
            fmap[i] = k.feature_params_map[i];
            theta2[i] = k.inv_lengthscales[fmap[i]].powi(2);
        }
    } else {
        let t2 = k.inv_lengthscales[0].powi(2);
        theta2.fill(t2);
    }

    let r: Vec<f64> = (0..nf).map(|i| f1[i] - f2[i]).collect();
    let mut d2 = 0.0;
    for i in 0..nf {
        d2 += theta2[i] * r[i] * r[i];
    }
    let sigma2 = k.signal_variance;
    let kval = sigma2 * (-d2).exp();

    let k_ee = kval;
    let u: Vec<f64> = (0..nf).map(|i| theta2[i] * r[i]).collect();

    let mut dk_df2 = vec![0.0; nf];
    let mut dk_df1 = vec![0.0; nf];
    for i in 0..nf {
        let v = 2.0 * kval * theta2[i] * r[i];
        dk_df2[i] = v;
        dk_df1[i] = -v;
    }

    let mut h_feat = Mat::<f64>::zeros(nf, nf);
    for i in 0..nf {
        h_feat[(i, i)] = 2.0 * kval * (theta2[i] - 2.0 * u[i] * u[i]);
        for j in (i + 1)..nf {
            let val = -4.0 * kval * u[i] * u[j];
            h_feat[(i, j)] = val;
            h_feat[(j, i)] = val;
        }
    }

    let k_ef = mat_t_vec(&j2, &dk_df2);
    let k_fe = mat_t_vec(&j1, &dk_df1);
    let k_ff = jt_h_j(&j1, &h_feat, &j2);

    let blocks = KernelBlocks { k_ee, k_ef, k_fe, k_ff };

    // Hyperparameter gradients
    let mut s_vec = vec![0.0; n_ls];
    for i in 0..nf {
        s_vec[fmap[i]] += r[i] * r[i];
    }
    for p in 0..n_ls {
        s_vec[p] *= 2.0 * k.inv_lengthscales[p].powi(2);
    }

    let mut grad_blocks = Vec::with_capacity(n_params);

    // d/d(log sigma2)
    grad_blocks.push(KernelBlocks {
        k_ee: blocks.k_ee,
        k_ef: blocks.k_ef.clone(),
        k_fe: blocks.k_fe.clone(),
        k_ff: blocks.k_ff.clone(),
    });

    for p in 0..n_ls {
        let sp = s_vec[p];
        let theta2_p = k.inv_lengthscales[p].powi(2);

        let dk_ee_p = -kval * sp;

        let mut ddk_df2_p = vec![0.0; nf];
        let mut ddk_df1_p = vec![0.0; nf];
        for l in 0..nf {
            let coeff = if fmap[l] == p { 2.0 * theta2_p } else { 0.0 } - theta2[l] * sp;
            let v = 2.0 * kval * r[l] * coeff;
            ddk_df2_p[l] = v;
            ddk_df1_p[l] = -v;
        }

        let dk_ef_p = mat_t_vec(&j2, &ddk_df2_p);
        let dk_fe_p = mat_t_vec(&j1, &ddk_df1_p);

        let mut du = vec![0.0; nf];
        for l in 0..nf {
            du[l] = if fmap[l] == p { 2.0 * theta2_p } else { 0.0 } * r[l];
        }

        let mut dh_feat = Mat::<f64>::zeros(nf, nf);
        for l in 0..nf {
            let diag_term = if fmap[l] == p { 2.0 * theta2_p } else { 0.0 };
            dh_feat[(l, l)] =
                -sp * h_feat[(l, l)] + 2.0 * kval * (diag_term - 4.0 * du[l] * u[l]);
            for m in (l + 1)..nf {
                let val =
                    -sp * h_feat[(l, m)] + 2.0 * kval * (-2.0 * (du[l] * u[m] + u[l] * du[m]));
                dh_feat[(l, m)] = val;
                dh_feat[(m, l)] = val;
            }
        }

        let dk_ff_p = jt_h_j(&j1, &dh_feat, &j2);

        grad_blocks.push(KernelBlocks {
            k_ee: dk_ee_p,
            k_ef: dk_ef_p,
            k_fe: dk_fe_p,
            k_ff: dk_ff_p,
        });
    }

    KernelBlocksWithGrads { blocks, grad_blocks }
}

/// Backward-compatible alias.
pub fn kernel_blocks_and_hypergrads(
    k: &MolInvDistSE,
    x1: &[f64],
    x2: &[f64],
) -> KernelBlocksWithGrads {
    molinvdist_kernel_blocks_and_hypergrads(k, x1, x2)
}

// ---------------------------------------------------------------------------
// CartesianSE kernel blocks (analytical, J = identity)
// ---------------------------------------------------------------------------

/// Compute kernel blocks for CartesianSE.
///
/// Features = coordinates, Jacobian = I. All matrix products simplify:
/// k_ef = dk_df2, k_fe = dk_df1, k_ff = H_feat.
pub fn cartesian_kernel_blocks(k: &CartesianSE, x1: &[f64], x2: &[f64]) -> KernelBlocks {
    let d = x1.len();
    let theta2 = k.inv_lengthscale * k.inv_lengthscale;

    let r: Vec<f64> = (0..d).map(|i| x1[i] - x2[i]).collect();
    let d2: f64 = r.iter().map(|v| v * v).sum::<f64>() * theta2;
    let kval = k.signal_variance * (-d2).exp();

    let k_ee = kval;

    // k_ef[i] = dk/dy_i = 2*theta^2*r_i*kval
    // k_fe[i] = dk/dx_i = -2*theta^2*r_i*kval
    let k_ef: Vec<f64> = r.iter().map(|&ri| 2.0 * theta2 * ri * kval).collect();
    let k_fe: Vec<f64> = r.iter().map(|&ri| -2.0 * theta2 * ri * kval).collect();

    // k_ff[i,j] = 2*theta^2*kval*(delta_ij - 2*theta^2*r_i*r_j)
    let mut k_ff = Mat::<f64>::zeros(d, d);
    for i in 0..d {
        k_ff[(i, i)] = 2.0 * theta2 * kval * (1.0 - 2.0 * theta2 * r[i] * r[i]);
        for j in (i + 1)..d {
            let val = -4.0 * theta2 * theta2 * kval * r[i] * r[j];
            k_ff[(i, j)] = val;
            k_ff[(j, i)] = val;
        }
    }

    KernelBlocks { k_ee, k_ef, k_fe, k_ff }
}

/// Compute kernel blocks AND hyperparameter gradients for CartesianSE.
///
/// Two parameters: [log(sigma2), log(theta)].
pub fn cartesian_kernel_blocks_and_hypergrads(
    k: &CartesianSE,
    x1: &[f64],
    x2: &[f64],
) -> KernelBlocksWithGrads {
    let d = x1.len();
    let theta2 = k.inv_lengthscale * k.inv_lengthscale;

    let r: Vec<f64> = (0..d).map(|i| x1[i] - x2[i]).collect();
    let r2_sum: f64 = r.iter().map(|v| v * v).sum();
    let d2 = theta2 * r2_sum;
    let kval = k.signal_variance * (-d2).exp();

    let k_ee = kval;

    let k_ef: Vec<f64> = r.iter().map(|&ri| 2.0 * theta2 * ri * kval).collect();
    let k_fe: Vec<f64> = r.iter().map(|&ri| -2.0 * theta2 * ri * kval).collect();

    let mut k_ff = Mat::<f64>::zeros(d, d);
    for i in 0..d {
        k_ff[(i, i)] = 2.0 * theta2 * kval * (1.0 - 2.0 * theta2 * r[i] * r[i]);
        for j in (i + 1)..d {
            let val = -4.0 * theta2 * theta2 * kval * r[i] * r[j];
            k_ff[(i, j)] = val;
            k_ff[(j, i)] = val;
        }
    }

    let blocks = KernelBlocks {
        k_ee,
        k_ef: k_ef.clone(),
        k_fe: k_fe.clone(),
        k_ff: k_ff.clone(),
    };

    let mut grad_blocks = Vec::with_capacity(2);

    // d/d(log sigma2): all blocks scale linearly with sigma2
    grad_blocks.push(KernelBlocks {
        k_ee: blocks.k_ee,
        k_ef: blocks.k_ef.clone(),
        k_fe: blocks.k_fe.clone(),
        k_ff: blocks.k_ff.clone(),
    });

    // d/d(log theta): sp = 2*theta^2 * sum r_i^2 = 2*d2
    let sp = 2.0 * d2;

    let dk_ee_theta = -kval * sp;

    // For CartesianSE isotropic: coeff = 2*theta^2 - theta^2*sp = theta^2*(2 - sp)
    let coeff = theta2 * (2.0 - sp);
    let dk_ef_theta: Vec<f64> = r.iter().map(|&ri| 2.0 * kval * ri * coeff).collect();
    let dk_fe_theta: Vec<f64> = r.iter().map(|&ri| -2.0 * kval * ri * coeff).collect();

    // u[i] = theta^2 * r[i], du[i] = 2*theta^2 * r[i]
    let mut dk_ff_theta = Mat::<f64>::zeros(d, d);
    for i in 0..d {
        let u_i = theta2 * r[i];
        let du_i = 2.0 * theta2 * r[i];
        dk_ff_theta[(i, i)] =
            -sp * k_ff[(i, i)] + 2.0 * kval * (2.0 * theta2 - 4.0 * du_i * u_i);
        for j in (i + 1)..d {
            let u_j = theta2 * r[j];
            let du_j = 2.0 * theta2 * r[j];
            let val = -sp * k_ff[(i, j)]
                + 2.0 * kval * (-2.0 * (du_i * u_j + u_i * du_j));
            dk_ff_theta[(i, j)] = val;
            dk_ff_theta[(j, i)] = val;
        }
    }

    grad_blocks.push(KernelBlocks {
        k_ee: dk_ee_theta,
        k_ef: dk_ef_theta,
        k_fe: dk_fe_theta,
        k_ff: dk_ff_theta,
    });

    KernelBlocksWithGrads { blocks, grad_blocks }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_self_eval() {
        let k = MolInvDistSE::isotropic(1.0, 0.5, vec![]);
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let val = k.eval(&x, &x);
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_kernel_blocks_symmetry() {
        let k = MolInvDistSE::isotropic(1.0, 0.5, vec![]);
        let x1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let x2 = vec![0.1, 0.0, 0.0, 1.1, 0.0, 0.0, 2.1, 0.0, 0.0];
        let b = kernel_blocks(&k, &x1, &x2);
        assert_eq!(b.k_ef.len(), 9);
        assert_eq!(b.k_fe.len(), 9);
        assert_eq!(b.k_ff.nrows(), 9);
        assert_eq!(b.k_ff.ncols(), 9);
        assert!((b.k_ee - k.eval(&x1, &x2)).abs() < 1e-12);
    }

    #[test]
    fn test_kernel_blocks_self_symmetry() {
        let k = MolInvDistSE::isotropic(2.0, 1.0, vec![]);
        let x = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let b = kernel_blocks(&k, &x, &x);
        for i in 0..b.k_ff.nrows() {
            for j in (i + 1)..b.k_ff.ncols() {
                assert!((b.k_ff[(i, j)] - b.k_ff[(j, i)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_cartesian_se_self_eval() {
        let k = CartesianSE::new(1.0, 0.5);
        let x = vec![0.5, 1.0];
        assert!((k.eval(&x, &x) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cartesian_se_blocks_symmetry() {
        let k = CartesianSE::new(2.0, 1.0);
        let x1 = vec![0.5, 1.0];
        let x2 = vec![0.7, 0.8];
        let b = cartesian_kernel_blocks(&k, &x1, &x2);
        assert_eq!(b.k_ef.len(), 2);
        assert_eq!(b.k_fe.len(), 2);
        assert_eq!(b.k_ff.nrows(), 2);
        // k_ef[d] = -k_fe[d] for SE kernels
        for d in 0..2 {
            assert!((b.k_ef[d] + b.k_fe[d]).abs() < 1e-12);
        }
        // k_ff should be symmetric
        assert!((b.k_ff[(0, 1)] - b.k_ff[(1, 0)]).abs() < 1e-12);
        // k_ee should match eval
        assert!((b.k_ee - k.eval(&x1, &x2)).abs() < 1e-12);
    }

    #[test]
    fn test_cartesian_se_blocks_self() {
        let k = CartesianSE::new(1.5, 2.0);
        let x = vec![0.3, 0.7];
        let b = cartesian_kernel_blocks(&k, &x, &x);
        // Self-eval: k_ee = sigma^2
        assert!((b.k_ee - 1.5).abs() < 1e-12);
        // k_ef = k_fe = 0 (r = 0)
        for d in 0..2 {
            assert!(b.k_ef[d].abs() < 1e-12);
            assert!(b.k_fe[d].abs() < 1e-12);
        }
        // k_ff diagonal = 2*theta^2*sigma^2
        let expected_diag = 2.0 * 4.0 * 1.5; // 2*theta^2*sigma^2
        for d in 0..2 {
            assert!((b.k_ff[(d, d)] - expected_diag).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cartesian_se_hypergrads_finite_diff() {
        let k = CartesianSE::new(1.5, 2.0);
        let x1 = vec![0.3, 0.7];
        let x2 = vec![0.5, 0.9];
        let bg = cartesian_kernel_blocks_and_hypergrads(&k, &x1, &x2);

        let eps = 1e-6;

        // Check d/d(log sigma2) via finite difference
        let k_plus = CartesianSE::new((1.5f64.ln() + eps).exp(), 2.0);
        let k_minus = CartesianSE::new((1.5f64.ln() - eps).exp(), 2.0);
        let b_plus = cartesian_kernel_blocks(&k_plus, &x1, &x2);
        let b_minus = cartesian_kernel_blocks(&k_minus, &x1, &x2);
        let fd_sigma = (b_plus.k_ee - b_minus.k_ee) / (2.0 * eps);
        assert!((bg.grad_blocks[0].k_ee - fd_sigma).abs() < 1e-4,
            "sigma2 grad: analytic={}, fd={}", bg.grad_blocks[0].k_ee, fd_sigma);

        // Check d/d(log theta) via finite difference
        let k_plus = CartesianSE::new(1.5, (2.0f64.ln() + eps).exp());
        let k_minus = CartesianSE::new(1.5, (2.0f64.ln() - eps).exp());
        let b_plus = cartesian_kernel_blocks(&k_plus, &x1, &x2);
        let b_minus = cartesian_kernel_blocks(&k_minus, &x1, &x2);
        let fd_theta = (b_plus.k_ee - b_minus.k_ee) / (2.0 * eps);
        assert!((bg.grad_blocks[1].k_ee - fd_theta).abs() < 1e-4,
            "theta grad: analytic={}, fd={}", bg.grad_blocks[1].k_ee, fd_theta);
    }

    #[test]
    fn test_kernel_enum_dispatch() {
        let k = Kernel::Cartesian(CartesianSE::new(1.0, 0.5));
        let x1 = vec![0.3, 0.7];
        let x2 = vec![0.5, 0.9];
        let b = k.kernel_blocks(&x1, &x2);
        assert_eq!(b.k_ef.len(), 2);
        assert!((b.k_ee - k.eval(&x1, &x2)).abs() < 1e-12);
    }
}
