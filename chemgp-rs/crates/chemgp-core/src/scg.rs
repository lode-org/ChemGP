//! Scaled Conjugate Gradient (SCG) optimizer -- Moller 1993.
//!
//! Ports `scg.jl`.

/// Result of SCG optimization.
pub struct ScgResult {
    pub w_best: Vec<f64>,
    pub f_best: f64,
    pub converged: bool,
}

/// SCG configuration.
pub struct ScgConfig {
    pub max_iter: usize,
    pub tol_x: f64,
    pub tol_f: f64,
    pub sigma0: f64,
    pub lambda_init: f64,
    pub lambda_max: f64,
    pub lambda_min: f64,
    pub verbose: bool,
}

impl Default for ScgConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol_x: 1e-6,
            tol_f: 1e-8,
            sigma0: 1e-4,
            lambda_init: 1.0,
            lambda_max: 1e100,
            lambda_min: 1e-15,
            verbose: false,
        }
    }
}

/// Minimize f(w) using Scaled Conjugate Gradient.
///
/// `fg` takes `w` and returns `(f_value, gradient)`.
pub fn scg_optimize<F>(fg: &mut F, w0: &[f64], config: &ScgConfig) -> ScgResult
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = w0.len();
    let mut w = w0.to_vec();

    let (f_old_val, r) = fg(&w);
    if !f_old_val.is_finite() {
        return ScgResult {
            w_best: w,
            f_best: f_old_val,
            converged: false,
        };
    }

    let mut r = r; // gradient
    let mut p: Vec<f64> = r.iter().map(|x| -x).collect(); // search direction

    let mut lambda = config.lambda_init;
    let mut success = true;
    let mut nsuccess = 0;
    let mut f_best = f_old_val;
    let mut f_old = f_old_val;
    let mut w_best = w.clone();
    let mut gamma = 0.0;
    let mut kappa = 0.0;
    let mut mu = 0.0;

    for _iter in 0..config.max_iter {
        if success {
            mu = dot(&p, &r);
            if mu >= 0.0 {
                p = r.iter().map(|x| -x).collect();
                mu = dot(&p, &r);
            }

            kappa = dot(&p, &p);
            if kappa < f64::EPSILON {
                return ScgResult {
                    w_best,
                    f_best,
                    converged: true,
                };
            }

            let sigma = config.sigma0 / kappa.sqrt();

            let w_new: Vec<f64> = w.iter().zip(p.iter()).map(|(wi, pi)| wi + sigma * pi).collect();
            let (f_new_val, g_plus) = fg(&w_new);
            if !f_new_val.is_finite() {
                lambda *= 4.0;
                success = false;
                continue;
            }

            gamma = dot(&p, &vsub(&g_plus, &r)) / sigma;
        }

        let mut delta = gamma + lambda * kappa;
        if delta <= 0.0 {
            delta = lambda * kappa;
            lambda -= gamma / kappa;
        }

        let alpha = -mu / delta;

        let w_new: Vec<f64> = w.iter().zip(p.iter()).map(|(wi, pi)| wi + alpha * pi).collect();
        let (f_new, g_new) = fg(&w_new);

        if !f_new.is_finite() {
            lambda *= 4.0;
            if lambda >= config.lambda_max {
                break;
            }
            success = false;
            continue;
        }

        let comparison = 2.0 * (f_new - f_old) / (alpha * mu);

        if comparison >= 0.0 {
            let f_prev = f_old;
            w = w_new;
            f_old = f_new;

            if f_new < f_best {
                f_best = f_new;
                w_best = w.clone();
            }

            // Convergence checks
            let max_step = alpha * kappa.sqrt();
            if max_step < config.tol_x {
                return ScgResult {
                    w_best,
                    f_best,
                    converged: true,
                };
            }
            if (f_new - f_prev).abs() < config.tol_f {
                return ScgResult {
                    w_best,
                    f_best,
                    converged: true,
                };
            }
            if g_new.iter().map(|x| x.abs()).fold(0.0f64, f64::max) < f64::EPSILON {
                return ScgResult {
                    w_best,
                    f_best,
                    converged: true,
                };
            }

            // Polak-Ribiere update
            let beta = (dot(&g_new, &g_new) - dot(&g_new, &r)) / (-mu);
            r = g_new;
            p = r.iter().zip(p.iter()).map(|(ri, pi)| -ri + beta * pi).collect();

            nsuccess += 1;
            if nsuccess >= n {
                p = r.iter().map(|x| -x).collect();
                nsuccess = 0;
            }

            success = true;
        } else {
            success = false;
        }

        if comparison < 0.25 {
            lambda = (lambda * 4.0).min(config.lambda_max);
        }
        if comparison > 0.75 {
            lambda = (lambda * 0.5).max(config.lambda_min);
        }

        if lambda >= config.lambda_max {
            break;
        }
    }

    ScgResult {
        w_best,
        f_best,
        converged: false,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vsub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scg_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let mut fg = |w: &[f64]| -> (f64, Vec<f64>) {
            let x = w[0];
            let y = w[1];
            let f = (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
            let gx = -2.0 * (1.0 - x) + 200.0 * (y - x * x) * (-2.0 * x);
            let gy = 200.0 * (y - x * x);
            (f, vec![gx, gy])
        };

        let config = ScgConfig {
            max_iter: 2000,
            tol_f: 1e-12,
            ..Default::default()
        };
        let result = scg_optimize(&mut fg, &[-1.0, 1.0], &config);
        assert!(result.f_best < 1e-4, "f_best = {}", result.f_best);
    }
}
