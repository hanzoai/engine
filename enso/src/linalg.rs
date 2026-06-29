//! The minimum dense `f64` linear algebra the policy needs: a bilinear form, a
//! Cholesky solve for the offline ridge fit, and Sherman-Morrison rank-1 inverse
//! updates for the online bandit. No external crate -- these are small, hot, and
//! exactly what is required, nothing more.

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out = m * v`, where `m` is `rows x cols` row-major.
pub fn matvec(m: &[f64], rows: usize, cols: usize, v: &[f64], out: &mut [f64]) {
    for i in 0..rows {
        out[i] = dot(&m[i * cols..i * cols + cols], v);
    }
}

/// The bilinear form `x^T W p` with `w` the `d x k` row-major weight matrix.
pub fn bilinear(x: &[f64], w: &[f64], p: &[f64], d: usize, k: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        s += x[i] * dot(&w[i * k..i * k + k], p);
    }
    s
}

/// The flattened bilinear feature `phi = vec(x p^T)`, so `dot(phi, vec(W)) ==
/// bilinear(x, W, p)`. This is the feature the linear learner operates on.
pub fn bilinear_feature(x: &[f64], p: &[f64], out: &mut [f64]) {
    let k = p.len();
    for (i, &xi) in x.iter().enumerate() {
        for (j, &pj) in p.iter().enumerate() {
            out[i * k + j] = xi * pj;
        }
    }
}

/// In-place Cholesky: overwrites the lower triangle of SPD `a` (`n x n`
/// row-major) with its factor `L` (`a = L L^T`). Inputs are made SPD by the
/// ridge term, so no pivoting is needed.
pub fn cholesky(a: &mut [f64], n: usize) {
    for j in 0..n {
        let mut d = a[j * n + j];
        for kk in 0..j {
            d -= a[j * n + kk] * a[j * n + kk];
        }
        let d = d.max(1e-12).sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for kk in 0..j {
                s -= a[i * n + kk] * a[j * n + kk];
            }
            a[i * n + j] = s / d;
        }
    }
}

/// Solve `L L^T x = b` given the Cholesky factor `l` from [`cholesky`].
pub fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for kk in 0..i {
            s -= l[i * n + kk] * y[kk];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for kk in (i + 1)..n {
            s -= l[kk * n + i] * x[kk];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Sherman-Morrison rank-1 update of an inverse: `ainv <- (A + phi phi^T)^-1`
/// given `ainv = A^-1` (`n x n`, symmetric). Returns `phi^T A^-1 phi` (the UCB
/// variance term) as a by-product, since it is computed on the way.
pub fn sherman_morrison(ainv: &mut [f64], n: usize, phi: &[f64]) -> f64 {
    let mut u = vec![0.0; n];
    matvec(ainv, n, n, phi, &mut u);
    let var = dot(phi, &u);
    let denom = 1.0 + var;
    for i in 0..n {
        let ui = u[i];
        let row = i * n;
        for j in 0..n {
            ainv[row + j] -= ui * u[j] / denom;
        }
    }
    var
}

/// The UCB variance term `phi^T A^-1 phi` without mutating `ainv`.
pub fn quad_form(ainv: &[f64], n: usize, phi: &[f64], scratch: &mut [f64]) -> f64 {
    matvec(ainv, n, n, phi, scratch);
    dot(phi, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_spd() {
        let mut a = vec![4.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        cholesky(&mut a, 2);
        let x = cholesky_solve(&a, 2, &b);
        // A*x should reproduce b (use the original A).
        assert!((4.0 * x[0] + 1.0 * x[1] - b[0]).abs() < 1e-9);
        assert!((1.0 * x[0] + 3.0 * x[1] - b[1]).abs() < 1e-9);
    }

    #[test]
    fn sherman_morrison_matches_closed_form() {
        // A = 2I (3x3) -> ainv = 0.5I; add phi=[1,1,0] -> (A+phi phi^T)^-1.
        let n = 3;
        let mut ainv = vec![0.0; n * n];
        for i in 0..n {
            ainv[i * n + i] = 0.5;
        }
        let phi = [1.0, 1.0, 0.0];
        let var = sherman_morrison(&mut ainv, n, &phi);
        assert!((var - 1.0).abs() < 1e-12); // phi^T (0.5I) phi = 0.5+0.5
        assert!((ainv[0] - 0.375).abs() < 1e-9);
        assert!((ainv[1] + 0.125).abs() < 1e-9);
        assert!((ainv[8] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn bilinear_equals_feature_dot() {
        let x = [1.0, 2.0];
        let p = [3.0, 4.0, 5.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut phi = vec![0.0; 6];
        bilinear_feature(&x, &p, &mut phi);
        assert!((bilinear(&x, &w, &p, 2, 3) - dot(&phi, &w)).abs() < 1e-12);
        assert!((bilinear(&x, &w, &p, 2, 3) - (1.0 * 3.0 + 2.0 * 4.0)).abs() < 1e-12);
    }
}
