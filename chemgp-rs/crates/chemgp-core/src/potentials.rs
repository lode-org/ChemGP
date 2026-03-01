//! Analytical potentials: LJ, Muller-Brown, LEPS.
//!
//! Pedagogical test surfaces for benchmarking GP-accelerated optimization.

/// Lennard-Jones energy and gradient for a flat coordinate vector.
pub fn lj_energy_gradient(x: &[f64], epsilon: f64, sigma: f64) -> (f64, Vec<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = vec![0.0; x.len()];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * j] - x[3 * i];
            let dy = x[3 * j + 1] - x[3 * i + 1];
            let dz = x[3 * j + 2] - x[3 * i + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();

            let sr6 = (sigma / r).powi(6);
            let sr12 = sr6 * sr6;

            e += 4.0 * epsilon * (sr12 - sr6);

            let f_over_r = (24.0 * epsilon / r2) * (2.0 * sr12 - sr6);

            for d in 0..3 {
                let rij_d = x[3 * j + d] - x[3 * i + d];
                g[3 * i + d] -= f_over_r * rij_d;
                g[3 * j + d] += f_over_r * rij_d;
            }
        }
    }

    (e, g)
}

// --- Muller-Brown ---

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

/// Known minima of the Muller-Brown surface.
pub const MULLER_BROWN_MINIMA: [[f64; 2]; 3] = [
    [-0.558224, 1.441726],
    [0.623499, 0.028038],
    [-0.050011, 0.466694],
];

/// Known saddle points.
pub const MULLER_BROWN_SADDLES: [[f64; 2]; 2] = [
    [-0.822002, 0.624313],
    [0.212487, 0.292988],
];

/// Muller-Brown potential: E and G for 2D point [x, y].
pub fn muller_brown_energy_gradient(xy: &[f64]) -> (f64, Vec<f64>) {
    let (x, y) = (xy[0], xy[1]);
    let mut e = 0.0;
    let mut dedx = 0.0;
    let mut dedy = 0.0;

    for k in 0..4 {
        let dx = x - MB_X0[k];
        let dy = y - MB_Y0[k];
        let exponent = MB_AA[k] * dx * dx + MB_B[k] * dx * dy + MB_C[k] * dy * dy;
        let term = MB_A[k] * exponent.exp();

        e += term;
        dedx += term * (2.0 * MB_AA[k] * dx + MB_B[k] * dy);
        dedy += term * (MB_B[k] * dx + 2.0 * MB_C[k] * dy);
    }

    (e, vec![dedx, dedy])
}

// --- LEPS ---

const LEPS_ALPHA: f64 = 1.942;
const LEPS_R_E: f64 = 0.742;
const LEPS_D_AB: f64 = 4.746;
const LEPS_D_BC: f64 = 4.746;
const LEPS_D_AC: f64 = 3.445;
const LEPS_S_AB: f64 = 0.05;
const LEPS_S_BC: f64 = 0.30;
const LEPS_S_AC: f64 = 0.05;

fn leps_q(r: f64, d: f64) -> (f64, f64) {
    let v = LEPS_ALPHA * (r - LEPS_R_E);
    let ev = (-v).exp();
    let e2v = (-2.0 * v).exp();
    let q = 0.5 * d * (1.5 * e2v - ev);
    let dq = 0.5 * d * LEPS_ALPHA * (-3.0 * e2v + ev);
    (q, dq)
}

fn leps_j(r: f64, d: f64) -> (f64, f64) {
    let v = LEPS_ALPHA * (r - LEPS_R_E);
    let ev = (-v).exp();
    let e2v = (-2.0 * v).exp();
    let j = 0.25 * d * (e2v - 6.0 * ev);
    let dj = 0.25 * d * LEPS_ALPHA * (6.0 * ev - 2.0 * e2v);
    (j, dj)
}

/// Reactant geometry: A far from B-C at equilibrium.
pub const LEPS_REACTANT: [f64; 9] = [
    0.0, 0.0, 0.0, LEPS_R_E, 0.0, 0.0, LEPS_R_E + 3.0, 0.0, 0.0,
];

/// Product geometry: A-B at equilibrium, C far away.
pub const LEPS_PRODUCT: [f64; 9] = [
    0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0 + LEPS_R_E, 0.0, 0.0,
];

/// LEPS potential for 3-atom system (9D flat coordinates).
pub fn leps_energy_gradient(x: &[f64]) -> (f64, Vec<f64>) {
    let ra = &x[0..3];
    let rb = &x[3..6];
    let rc = &x[6..9];

    let d_ab = [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]];
    let d_bc = [rc[0] - rb[0], rc[1] - rb[1], rc[2] - rb[2]];
    let d_ac = [rc[0] - ra[0], rc[1] - ra[1], rc[2] - ra[2]];

    let r_ab = (d_ab[0] * d_ab[0] + d_ab[1] * d_ab[1] + d_ab[2] * d_ab[2]).sqrt();
    let r_bc = (d_bc[0] * d_bc[0] + d_bc[1] * d_bc[1] + d_bc[2] * d_bc[2]).sqrt();
    let r_ac = (d_ac[0] * d_ac[0] + d_ac[1] * d_ac[1] + d_ac[2] * d_ac[2]).sqrt();

    let u_ab: [f64; 3] = if r_ab > 1e-12 {
        [d_ab[0] / r_ab, d_ab[1] / r_ab, d_ab[2] / r_ab]
    } else {
        [0.0; 3]
    };
    let u_bc: [f64; 3] = if r_bc > 1e-12 {
        [d_bc[0] / r_bc, d_bc[1] / r_bc, d_bc[2] / r_bc]
    } else {
        [0.0; 3]
    };
    let u_ac: [f64; 3] = if r_ac > 1e-12 {
        [d_ac[0] / r_ac, d_ac[1] / r_ac, d_ac[2] / r_ac]
    } else {
        [0.0; 3]
    };

    let op_ab = 1.0 / (1.0 + LEPS_S_AB);
    let op_bc = 1.0 / (1.0 + LEPS_S_BC);
    let op_ac = 1.0 / (1.0 + LEPS_S_AC);

    let (q_ab, dq_ab) = leps_q(r_ab, LEPS_D_AB);
    let (q_bc, dq_bc) = leps_q(r_bc, LEPS_D_BC);
    let (q_ac, dq_ac) = leps_q(r_ac, LEPS_D_AC);

    let (j_ab, dj_ab) = leps_j(r_ab, LEPS_D_AB);
    let (j_bc, dj_bc) = leps_j(r_bc, LEPS_D_BC);
    let (j_ac, dj_ac) = leps_j(r_ac, LEPS_D_AC);

    let qs = q_ab * op_ab + q_bc * op_bc + q_ac * op_ac;
    let j_ab_s = j_ab * op_ab;
    let j_bc_s = j_bc * op_bc;
    let j_ac_s = j_ac * op_ac;

    let js = j_ab_s * j_ab_s + j_bc_s * j_bc_s + j_ac_s * j_ac_s
        - j_ab_s * j_bc_s
        - j_bc_s * j_ac_s
        - j_ab_s * j_ac_s;
    let sqrt_j = js.max(1e-30).sqrt();

    let e = qs - sqrt_j;

    // Gradient
    let dqs_d_ab = dq_ab * op_ab;
    let dqs_d_bc = dq_bc * op_bc;
    let dqs_d_ac = dq_ac * op_ac;

    let dj_ab_s = dj_ab * op_ab;
    let dj_bc_s = dj_bc * op_bc;
    let dj_ac_s = dj_ac * op_ac;

    let djs_d_ab = dj_ab_s * (2.0 * j_ab_s - j_bc_s - j_ac_s);
    let djs_d_bc = dj_bc_s * (2.0 * j_bc_s - j_ab_s - j_ac_s);
    let djs_d_ac = dj_ac_s * (2.0 * j_ac_s - j_bc_s - j_ab_s);

    let de_d_ab = dqs_d_ab - 0.5 / sqrt_j * djs_d_ab;
    let de_d_bc = dqs_d_bc - 0.5 / sqrt_j * djs_d_bc;
    let de_d_ac = dqs_d_ac - 0.5 / sqrt_j * djs_d_ac;

    let mut g = vec![0.0; 9];

    // Atom A (0..3)
    for k in 0..3 {
        g[k] = -de_d_ab * u_ab[k] - de_d_ac * u_ac[k];
    }
    // Atom B (3..6)
    for k in 0..3 {
        g[3 + k] = de_d_ab * u_ab[k] - de_d_bc * u_bc[k];
    }
    // Atom C (6..9)
    for k in 0..3 {
        g[6 + k] = de_d_bc * u_bc[k] + de_d_ac * u_ac[k];
    }

    (e, g)
}

/// LEPS in 2D reduced coordinates [r_AB, r_BC].
pub fn leps_energy_gradient_2d(r_ab_bc: &[f64]) -> (f64, Vec<f64>) {
    let r_ab = r_ab_bc[0];
    let r_bc = r_ab_bc[1];
    let r_ac = r_ab + r_bc;

    let op_ab = 1.0 / (1.0 + LEPS_S_AB);
    let op_bc = 1.0 / (1.0 + LEPS_S_BC);
    let op_ac = 1.0 / (1.0 + LEPS_S_AC);

    let (q_ab, dq_ab) = leps_q(r_ab, LEPS_D_AB);
    let (q_bc, dq_bc) = leps_q(r_bc, LEPS_D_BC);
    let (q_ac, dq_ac) = leps_q(r_ac, LEPS_D_AC);

    let (j_ab, dj_ab) = leps_j(r_ab, LEPS_D_AB);
    let (j_bc, dj_bc) = leps_j(r_bc, LEPS_D_BC);
    let (j_ac, dj_ac) = leps_j(r_ac, LEPS_D_AC);

    let qs = q_ab * op_ab + q_bc * op_bc + q_ac * op_ac;
    let j_ab_s = j_ab * op_ab;
    let j_bc_s = j_bc * op_bc;
    let j_ac_s = j_ac * op_ac;

    let js = j_ab_s * j_ab_s + j_bc_s * j_bc_s + j_ac_s * j_ac_s
        - j_ab_s * j_bc_s
        - j_bc_s * j_ac_s
        - j_ab_s * j_ac_s;
    let sqrt_j = js.max(1e-30).sqrt();

    let e = qs - sqrt_j;

    let dj_ab_s = dj_ab * op_ab;
    let dj_bc_s = dj_bc * op_bc;
    let dj_ac_s = dj_ac * op_ac;

    let djs_d_ab = dj_ab_s * (2.0 * j_ab_s - j_bc_s - j_ac_s);
    let djs_d_bc = dj_bc_s * (2.0 * j_bc_s - j_ab_s - j_ac_s);
    let djs_d_ac = dj_ac_s * (2.0 * j_ac_s - j_bc_s - j_ab_s);

    let de_d_ab = dq_ab * op_ab - 0.5 / sqrt_j * djs_d_ab;
    let de_d_bc = dq_bc * op_bc - 0.5 / sqrt_j * djs_d_bc;
    let de_d_ac = dq_ac * op_ac - 0.5 / sqrt_j * djs_d_ac;

    // Chain rule: r_AC = r_AB + r_BC
    let g_r_ab = de_d_ab + de_d_ac;
    let g_r_bc = de_d_bc + de_d_ac;

    (e, vec![g_r_ab, g_r_bc])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lj_minimum() {
        // Two atoms at sigma*2^(1/6) apart (LJ minimum)
        let r_min = 2.0f64.powf(1.0 / 6.0);
        let x = vec![0.0, 0.0, 0.0, r_min, 0.0, 0.0];
        let (e, g) = lj_energy_gradient(&x, 1.0, 1.0);
        assert_relative_eq!(e, -1.0, epsilon = 1e-10);
        assert!(g.iter().map(|x| x.abs()).sum::<f64>() < 1e-10);
    }

    #[test]
    fn test_muller_brown_minimum() {
        let (e, _) = muller_brown_energy_gradient(&MULLER_BROWN_MINIMA[0]);
        assert!(e < -140.0);
    }

    #[test]
    fn test_leps_reactant() {
        let (e, g) = leps_energy_gradient(&LEPS_REACTANT);
        assert!(e < 0.0, "LEPS reactant energy should be negative, got {}", e);
        // Forces should be small at near-equilibrium
        let max_f: f64 = g.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_f < 5.0, "LEPS reactant max force too large: {}", max_f);
    }
}
