/*
NN Chapter 04. Embedding Lookup and a Shared Convolution
[src/bin/nn_04_representations.rs]

Goal:
Gather token embeddings, slide a shared two-position filter, and backpropagate
through repeated token lookups and the filter weights.

Why this matters for neural networks:
Token IDs are addresses, not numerical measurements of meaning. An embedding
selects a row of learned parameters. A convolution reuses the same local rule
at each position instead of assigning an unrelated rule to every location.

Setup and dimensions:
E: V x D embedding table; IDs: T; X=lookup(E,IDs): T x D.
K: R x D, one filter; bias: scalar; output: T-R+1.
This exercise uses D=2 and R=2; stride 1, valid padding, no dilation.

Equations and gradient rules:
X[t][j] = E[ids[t]][j]
z[t] = b + sum_r sum_j K[r][j]*X[t+r][j]
a[t] = max(0,z[t])
Given an incoming gradient da[t]:
dz[t] = da[t] if z[t]>0, else 0
dK[r][j] = sum_t dz[t]*X[t+r][j]
dX[t+r][j] += dz[t]*K[r][j]; db = sum_t dz[t]
dE[id][j] += dX[t][j] for every t whose id equals id.
Repeated IDs must ACCUMULATE, not overwrite, their gradient.

Implementation steps:
1. Bounds-check IDs before indexing the table.
2. Reuse K at every valid window; do not advance a separate weight matrix.
3. Cache preactivations to choose the ReLU derivative.
4. Scatter-add input gradients back into the embedding table.

Important limitations:
The ReLU derivative at zero is chosen as 0. Numerical gradient tests avoid that
kink. We differentiate an explicit upstream objective sum_t upstream[t]*a[t],
not a fictional classification task. Parameters here are fixed, not trained.

The jump to the next idea:
A local filter has a bounded receptive field. A recurrent state carries a summary
across arbitrarily many steps, introducing the temporal gradient problem.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_04_representations
  cargo test --bin nn_04_representations
Or, independently:
  rustc --edition=2021 nn_04_representations.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://www.jmlr.org/papers/v3/bengio03a.html
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

type Row = [f64; 2];
fn lookup(e: &[Row], ids: &[usize]) -> Vec<Row> {
    assert!(!e.is_empty());
    ids.iter()
        .map(|&i| {
            assert!(i < e.len());
            e[i]
        })
        .collect()
}
/// Hints: output length is T-R+1; K has no time index; add b once per window.
fn conv(x: &[Row], k: &[Row; 2], b: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(x.len() >= 2);
    let mut z = Vec::new();
    for t in 0..x.len() - 1 {
        let mut value = b;
        for r in 0..2 {
            for j in 0..2 {
                value += k[r][j] * x[t + r][j]
            }
        }
        z.push(value);
    }
    let a = z.iter().map(|v| v.max(0.0)).collect();
    (z, a)
}
/// Hints: overlapping windows both contribute to dX; reset gradients to zero.
/// Mask the incoming gradient at nonpositive preactivations before accumulating.
fn conv_backward(x: &[Row], k: &[Row; 2], b: f64, upstream: &[f64]) -> ([Row; 2], f64, Vec<Row>) {
    let (z, _) = conv(x, k, b);
    assert_eq!(z.len(), upstream.len());
    let (mut dk, mut db, mut dx) = ([[0.0; 2]; 2], 0.0, vec![[0.0; 2]; x.len()]);
    for t in 0..z.len() {
        let dz = if z[t] > 0.0 { upstream[t] } else { 0.0 };
        db += dz;
        for r in 0..2 {
            for j in 0..2 {
                dk[r][j] += dz * x[t + r][j];
                dx[t + r][j] += dz * k[r][j]
            }
        }
    }
    (dk, db, dx)
}
fn scatter(v: usize, ids: &[usize], dx: &[Row]) -> Vec<Row> {
    assert_eq!(ids.len(), dx.len());
    let mut de = vec![[0.0; 2]; v];
    for (t, &id) in ids.iter().enumerate() {
        assert!(id < v);
        for j in 0..2 {
            de[id][j] += dx[t][j]
        }
    }
    de
}

fn cases() -> Vec<Case> {
    vec![
        ("Lookup preserves order and repeats", || {
            assert_eq!(
                lookup(&[[1.0, 2.0], [3.0, 4.0]], &[1, 0, 1]),
                vec![[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]
            )
        }),
        ("Worked convolution gives [1.5,0]", || {
            assert_eq!(
                conv(
                    &[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    &[[1.0, -1.0], [0.5, 0.5]],
                    0.0
                )
                .1,
                vec![1.5, 0.0]
            )
        }),
        ("Repeated IDs scatter-add gradients", || {
            assert_eq!(
                scatter(2, &[1, 1], &[[1.0, 2.0], [3.0, 4.0]]),
                vec![[0.0, 0.0], [4.0, 6.0]]
            )
        }),
        ("All four filter gradients match finite differences", || {
            let x = [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]];
            let k = [[0.1, 0.2], [0.3, 0.4]];
            let u = [0.7, -0.2];
            let (dk, _, _) = conv_backward(&x, &k, 1.0, &u);
            let objective = |k: &[Row; 2]| {
                conv(&x, k, 1.0)
                    .1
                    .iter()
                    .zip(u)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            };
            for r in 0..2 {
                for j in 0..2 {
                    let (mut a, mut b) = (k, k);
                    a[r][j] += 1e-6;
                    b[r][j] -= 1e-6;
                    close(dk[r][j], (objective(&a) - objective(&b)) / 2e-6, 1e-8)
                }
            }
        }),
        ("Overlapping windows add into the middle input", || {
            let (_, db, dx) = conv_backward(&[[1.0; 2]; 3], &[[1.0; 2]; 2], 0.0, &[1.0, 1.0]);
            close(db, 2.0, 1e-12);
            assert_eq!(dx, vec![[1.0; 2], [2.0; 2], [1.0; 2]]);
        }),
        ("Inactive ReLU blocks the gradient", || {
            let (dk, db, dx) = conv_backward(&[[1.0; 2]; 2], &[[1.0; 2]; 2], -10.0, &[1.0]);
            assert_eq!(dk, [[0.0; 2]; 2]);
            close(db, 0.0, 0.0);
            assert_eq!(dx, vec![[0.0; 2]; 2]);
        }),
    ]
}
fn demo() {
    let e = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let ids = [0, 1, 2, 1];
    let x = lookup(&e, &ids);
    let k = [[1.0, -1.0], [0.5, 0.5]];
    let (z, a) = conv(&x, &k, 0.0);
    let (dk, db, dx) = conv_backward(&x, &k, 0.0, &vec![1.0; a.len()]);
    println!(
        "IDs={ids:?}\nEmbeddings={x:?}\nz={z:?}\na={a:?}\ndK={dk:?}, db={db}\ndE={:?}",
        scatter(e.len(), &ids, &dx)
    );
}
fn main() {
    println!("NN Chapter 04: Embedding Lookup and a Shared Convolution");
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
