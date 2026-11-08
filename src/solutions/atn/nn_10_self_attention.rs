/*
NN Chapter 10. Scaled Dot-Product Attention, Forward and Backward
[src/bin/nn_10_self_attention.rs]

Goal:
Implement matrix attention with optional causal masking, then derive and test
all query, key and value gradients for a scalar downstream objective.

Why this matters for neural networks:
Additive attention retrieves source states using a query. Dot-product attention
organizes many queries at once. Q and K determine routing; V supplies the content.
The mask changes which information is available, not just how it is displayed.

Setup and dimensions:
Q: T x dk, K: S x dk, V: S x dv.
Scores and probabilities A: T x S; output O: T x dv.
For causal self-attention T=S; position i may attend to j<=i.
This exercise accepts Q/K/V directly; learned projections can wrap the kernel.

Equations and gradient rules:
S=Q K^T / sqrt(dk); masked S_ij=-infinity when j>i
A=row_softmax(S); O=A V
Given dO: dV=A^T dO; dA=dO V^T
dS_ij=A_ij*(dA_ij-sum_k A_ik*dA_ik)
dQ=dS K/sqrt(dk); dK=dS^T Q/sqrt(dk)
Masked scores have zero derivative because their A entries are exactly zero.
If Q=XWq, then dWq=X^T dQ and dX gets dQ Wq^T plus the K/V routes.

Implementation steps:
1. Check rectangular shapes before contracting dimensions.
2. Scale scores, mask future positions, then apply row-softmax.
3. Backpropagate through AV and each row's softmax independently.
4. Accumulate the Q/K/V paths. Use distinct test matrices, not Q=K=V everywhere.

Important limitations:
Every row must have at least one finite score; a fully masked row has no defined
probability distribution and is rejected. The causal mask assumes aligned query
and key positions starting at zero, not an offset KV-cache decoding API.
This exercise is a differentiable kernel, not a trained language model.

The jump to the next idea:
A Transformer adds learned projections, multiple heads, residual paths,
normalization, positional information and a positionwise MLP around this kernel.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_10_self_attention
  cargo test --bin nn_10_self_attention
Or, independently:
  rustc --edition=2021 nn_10_self_attention.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://arxiv.org/html/1706.03762
*/

use std::io::IsTerminal;

// These are correctness checks, NOT a dataset accuracy or a model-quality score.
// Each named case may contain several assertions. Count all cases, not just
// displayed predictions. No sleeps, external crates, network or random seed.
type Case = (&'static str, fn());
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "non-finite value: actual={actual}, expected={expected}"
    );
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.10}, expected={expected:.10}, tolerance={tolerance}"
    );
}
fn run_cases(cases: Vec<Case>) {
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let mut passed = 0;
    println!("\nCORRECTNESS TESTS (not a fit score)");
    for (name, test) in &cases {
        let ok = std::panic::catch_unwind(*test).is_ok();
        if ok {
            passed += 1;
        }
        let status = if ok { "PASS" } else { "FAIL" };
        let ink = if !color {
            ""
        } else if ok {
            "\x1b[32m"
        } else {
            "\x1b[31m"
        };
        let reset = if color { "\x1b[0m" } else { "" };
        println!("  {ink}{status}{reset}  {name}");
    }
    println!("Summary: {passed}/{} cases passed", cases.len());
    assert_eq!(passed, cases.len(), "Correctness suite failed");
}

type Mat = Vec<Vec<f64>>;
/// Hints: A is m x k and B is k x n. Contract k; preserve m and n. Check rectangularity before indexing.
fn matmul(a: &Mat, b: &Mat) -> Mat {
    assert!(!a.is_empty() && !b.is_empty() && !a[0].is_empty() && !b[0].is_empty());
    assert!(a.iter().all(|r| r.len() == a[0].len()));
    assert!(b.iter().all(|r| r.len() == b[0].len()));
    assert_eq!(a[0].len(), b.len());
    (0..a.len())
        .map(|i| {
            (0..b[0].len())
                .map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}
/// Hints: Output row j is input column j. Transposition changes layout, not the underlying values.
fn transpose(a: &Mat) -> Mat {
    (0..a[0].len())
        .map(|j| (0..a.len()).map(|i| a[i][j]).collect())
        .collect()
}
/// Hints: Subtract the largest allowed score, exponentiate, then normalize. A masked -infinity becomes probability zero.
fn softmax(row: &[f64]) -> Vec<f64> {
    assert!(!row.is_empty() && row.iter().any(|x| x.is_finite()));
    assert!(row.iter().all(|x| x.is_finite() || *x == f64::NEG_INFINITY));
    let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|x| x / sum).collect()
}
/// Hints: Compute QK^T, divide by sqrt(dk), mask BEFORE softmax, then multiply normalized rows by V.
fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> (Mat, Mat) {
    assert_eq!(k.len(), v.len());
    assert!(!q.is_empty() && !k.is_empty());
    assert!(k.iter().all(|r| r.len() == q[0].len()));
    let scores = matmul(q, &transpose(k));
    let scale = (q[0].len() as f64).sqrt();
    let a: Mat = scores
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let s: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(j, x)| {
                    if causal && j > i {
                        f64::NEG_INFINITY
                    } else {
                        x / scale
                    }
                })
                .collect();
            softmax(&s)
        })
        .collect();
    (matmul(&a, v), a)
}

/// Hints: upstream has output shape T x dv. Keep dA and dS separate.
/// The scale appears in both dQ and dK, but not in dV.
fn backward(q: &Mat, k: &Mat, v: &Mat, up: &Mat, causal: bool) -> (Mat, Mat, Mat) {
    let (_, a) = attention(q, k, v, causal);
    let dv = matmul(&transpose(&a), up);
    let da = matmul(up, &transpose(v));
    let mut ds = a.clone();
    for i in 0..a.len() {
        let mean: f64 = a[i].iter().zip(&da[i]).map(|(p, g)| p * g).sum();
        for j in 0..a[i].len() {
            ds[i][j] = a[i][j] * (da[i][j] - mean)
        }
    }
    let scale = (q[0].len() as f64).sqrt();
    let mut dq = matmul(&ds, k);
    let mut dk = matmul(&transpose(&ds), q);
    for m in [&mut dq, &mut dk] {
        for row in m {
            for x in row {
                *x /= scale
            }
        }
    }
    (dq, dk, dv)
}
fn fixture() -> (Mat, Mat, Mat, Mat) {
    (
        vec![vec![0.2, -0.3], vec![0.5, 0.7]],
        vec![vec![0.8, 0.1], vec![-0.4, 0.6]],
        vec![vec![0.3, 0.9], vec![0.7, -0.2]],
        vec![vec![0.6, -0.1], vec![0.2, 0.5]],
    )
}
fn objective(q: &Mat, k: &Mat, v: &Mat, up: &Mat, causal: bool) -> f64 {
    attention(q, k, v, causal)
        .0
        .iter()
        .zip(up)
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>())
        .sum()
}
fn check_gradients(causal: bool) {
    let (q, k, v, up) = fixture();
    let (dq, dk, dv) = backward(&q, &k, &v, &up, causal);
    let inputs = [q, k, v];
    let gradients = [dq, dk, dv];
    for which in 0..3 {
        for i in 0..2 {
            for j in 0..2 {
                let (mut a, mut b) = (inputs.clone(), inputs.clone());
                a[which][i][j] += 1e-6;
                b[which][i][j] -= 1e-6;
                let numerical = (objective(&a[0], &a[1], &a[2], &up, causal)
                    - objective(&b[0], &b[1], &b[2], &up, causal))
                    / 2e-6;
                close(gradients[which][i][j], numerical, 1e-8);
            }
        }
    }
}

fn cases() -> Vec<Case> {
    vec![
        ("Rectangular matrix multiplication", || {
            assert_eq!(
                matmul(
                    &vec![vec![1.0, 2.0, 3.0]],
                    &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 3.0]]
                ),
                vec![vec![7.0, 11.0]]
            )
        }),
        ("Each attention row is normalized", || {
            let (q, k, v, _) = fixture();
            for row in attention(&q, &k, &v, false).1 {
                close(row.iter().sum(), 1.0, 1e-12)
            }
        }),
        ("Causal first row selects first value exactly", || {
            let (q, k, v, _) = fixture();
            let (out, a) = attention(&q, &k, &v, true);
            close(a[0][1], 0.0, 0.0);
            assert_eq!(out[0], v[0]);
        }),
        ("All Q/K/V derivatives without mask", || {
            check_gradients(false)
        }),
        ("All Q/K/V derivatives with causal mask", || {
            check_gradients(true)
        }),
        ("Future value changes cannot alter earlier output", || {
            let (q, k, mut v, _) = fixture();
            let before = attention(&q, &k, &v, true).0;
            v[1] = vec![900.0, -900.0];
            assert_eq!(attention(&q, &k, &v, true).0[0], before[0]);
        }),
        ("Zero Q/K gives uniform average of values", || {
            let q = vec![vec![0.0; 2]; 2];
            let v = vec![vec![2.0], vec![4.0]];
            for row in attention(&q, &q, &v, false).0 {
                close(row[0], 3.0, 1e-12)
            }
        }),
    ]
}
fn demo() {
    let (q, k, v, up) = fixture();
    for causal in [false, true] {
        let (o, a) = attention(&q, &k, &v, causal);
        let (dq, dk, dv) = backward(&q, &k, &v, &up, causal);
        println!("causal={causal}\nA={a:.5?}\nO={o:.5?}\ndQ={dq:.5?}\ndK={dk:.5?}\ndV={dv:.5?}");
    }
}
fn main() {
    println!("NN Chapter 10: Scaled Dot-Product Attention, Forward and Backward");
    run_cases(cases());
    println!("\nDEMONSTRATION");
    demo();
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correctness_suite() {
        run_cases(cases());
    }
}
