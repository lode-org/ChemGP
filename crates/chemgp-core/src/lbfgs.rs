//! L-BFGS two-loop recursion.
//!
//! Ports `lbfgs.jl`.

/// Circular buffer for L-BFGS step/gradient-difference pairs.
#[derive(Debug, Clone)]
pub struct LbfgsHistory {
    pub m: usize,
    pub s: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
    pub count: usize,
}

impl LbfgsHistory {
    pub fn new(m: usize) -> Self {
        Self {
            m,
            s: Vec::new(),
            y: Vec::new(),
            count: 0,
        }
    }

    /// Push a (s, y) pair. Skipped if curvature condition y'*s <= 0.
    pub fn push_pair(&mut self, s: Vec<f64>, y: Vec<f64>) {
        let ys: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        if ys <= 1e-18 {
            return;
        }

        if self.s.len() >= self.m {
            self.s.remove(0);
            self.y.remove(0);
        }

        self.s.push(s);
        self.y.push(y);
        self.count += 1;
    }

    /// Compute L-BFGS search direction d = -H_k * gradient.
    pub fn compute_direction(&self, gradient: &[f64]) -> Vec<f64> {
        let m = self.s.len();

        if m == 0 {
            return gradient.iter().map(|x| -x).collect();
        }

        let mut q = gradient.to_vec();
        let mut alpha_vec = vec![0.0; m];
        let mut rho = vec![0.0; m];

        for i in 0..m {
            let ys: f64 = self.y[i].iter().zip(self.s[i].iter()).map(|(a, b)| a * b).sum();
            rho[i] = if ys > 1e-18 { 1.0 / ys } else { 0.0 };
        }

        // Backward pass
        for i in (0..m).rev() {
            alpha_vec[i] = rho[i] * dot(&self.s[i], &q);
            for j in 0..q.len() {
                q[j] -= alpha_vec[i] * self.y[i][j];
            }
        }

        // Initial Hessian scaling
        let s_last = &self.s[m - 1];
        let y_last = &self.y[m - 1];
        let yy = dot(y_last, y_last);
        let gamma = if yy > 1e-18 {
            dot(s_last, y_last) / yy
        } else {
            1.0
        };

        let mut r: Vec<f64> = q.iter().map(|x| gamma * x).collect();

        // Forward pass
        for i in 0..m {
            let beta = rho[i] * dot(&self.y[i], &r);
            for j in 0..r.len() {
                r[j] += (alpha_vec[i] - beta) * self.s[i][j];
            }
        }

        r.iter().map(|x| -x).collect()
    }

    pub fn reset(&mut self) {
        self.s.clear();
        self.y.clear();
        self.count = 0;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgs_steepest_descent_fallback() {
        let h = LbfgsHistory::new(5);
        let g = vec![1.0, -2.0, 3.0];
        let d = h.compute_direction(&g);
        assert_eq!(d, vec![-1.0, 2.0, -3.0]);
    }
}
