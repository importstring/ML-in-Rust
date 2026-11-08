/*
NN Chapter 09. Additive Attention and Query Gradients
[src/bin/nn_09_attention.rs]

Goal:
Score encoder states against a decoder query, normalize with stable softmax,
retrieve a weighted context, and differentiate a context loss back into the query.

Why this matters for neural networks:
The decoder no longer has to use the same fixed source summary at every step.
Its query changes the retrieval weights. The weighted sum is differentiable,
so a downstream loss can teach the scoring mechanism where to attend.

Setup and dimensions:
Query q: 2; encoder states H: N x 2; score vector and weights: N; context: 2.
This minimal additive scorer fixes Wq=Wh=identity and v=[1,-0.5].
The exercise differentiates q while holding H and the scorer parameters fixed.

Equations and gradient rules:
e_j=sum_k v_k*tanh(q_k+H[j][k])
a_j=exp(e_j-max(e))/sum_l exp(e_l-max(e))
c_k=sum_j a_j*H[j][k]
L=0.5*sum_k(c_k-target_k)^2; dc=c-target
da_j=sum_k dc_k*H[j][k]
de_j=a_j*(da_j-sum_l a_l*da_l)
dq_k=sum_j de_j*v_k*(1-tanh(q_k+H[j][k])^2)
The subtraction in de accounts for every softmax output depending on every score.

Implementation steps:
1. Compute one scalar score for each source state.
2. Subtract the largest score before exponentiation.
3. Normalize across source positions, not feature coordinates.
4. Backpropagate through the context sum, softmax, then additive scorer.

Important limitations:
This simplified scorer is not the full trainable Bahdanau parameterization.
Attention weights sum to one but should not automatically be treated as a
causal explanation of a model's decision. No translator is trained here.

The jump to the next idea:
Self-attention obtains queries, keys and values from the same sequence and
computes all query-position retrievals as matrix operations.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_09_attention
  cargo test --bin nn_09_attention
Or, independently:
  rustc --edition=2021 nn_09_attention.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html
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

/// Hints: Subtract the largest allowed score, exponentiate, then normalize. A masked -infinity becomes probability zero.
fn softmax(scores: &[f64]) -> Vec<f64> {
    assert!(!scores.is_empty() && scores.iter().all(|x| x.is_finite()));
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<_> = scores.iter().map(|x| (x - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|x| x / sum).collect()
}
/// Hints: scores have length N; context has length D. Do not mix these axes.
fn attend(q: [f64; 2], h: &[[f64; 2]]) -> (Vec<f64>, [f64; 2]) {
    let e: Vec<_> = h
        .iter()
        .map(|s| (q[0] + s[0]).tanh() - 0.5 * (q[1] + s[1]).tanh())
        .collect();
    let a = softmax(&e);
    let mut c = [0.0; 2];
    for (j, s) in h.iter().enumerate() {
        for k in 0..2 {
            c[k] += a[j] * s[k]
        }
    }
    (a, c)
}
fn loss(q: [f64; 2], h: &[[f64; 2]], y: [f64; 2]) -> f64 {
    let (_, c) = attend(q, h);
    0.5 * ((c[0] - y[0]).powi(2) + (c[1] - y[1]).powi(2))
}
/// Hints: softmax backward is a Jacobian-vector product; do not use a_j*(1-a_j)
/// alone. That diagonal-only shortcut loses cross-position effects.
fn query_gradient(q: [f64; 2], h: &[[f64; 2]], y: [f64; 2]) -> [f64; 2] {
    let (a, c) = attend(q, h);
    let dc = [c[0] - y[0], c[1] - y[1]];
    let da: Vec<_> = h.iter().map(|s| dc[0] * s[0] + dc[1] * s[1]).collect();
    let mean: f64 = a.iter().zip(&da).map(|(a, d)| a * d).sum();
    let mut dq = [0.0; 2];
    for j in 0..h.len() {
        let de = a[j] * (da[j] - mean);
        for k in 0..2 {
            dq[k] += de * [1.0, -0.5][k] * (1.0 - (q[k] + h[j][k]).tanh().powi(2))
        }
    }
    dq
}
fn states() -> Vec<[f64; 2]> {
    vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
}

fn cases() -> Vec<Case> {
    vec![
        ("Softmax sums to one with large scores", || {
            close(softmax(&[1000.0, 1001.0, 999.0]).iter().sum(), 1.0, 1e-12)
        }),
        ("Common score shift changes no weights", || {
            for (a, b) in softmax(&[1.0, 2.0, 3.0])
                .iter()
                .zip(softmax(&[1001.0, 1002.0, 1003.0]))
            {
                close(*a, b, 1e-12)
            }
        }),
        ("Equal scores give uniform weights", || {
            for a in softmax(&[4.0; 3]) {
                close(a, 1.0 / 3.0, 1e-12)
            }
        }),
        ("Worked context matches retrieval example", || {
            let (_, c) = attend([0.5, -0.25], &states());
            close(c[0], 0.799024, 1e-6);
            close(c[1], 0.513980, 1e-6)
        }),
        ("Both query gradients match finite differences", || {
            let q = [0.5, -0.25];
            let h = states();
            let y = [0.3, 0.8];
            let g = query_gradient(q, &h, y);
            for k in 0..2 {
                let (mut a, mut b) = (q, q);
                a[k] += 1e-6;
                b[k] -= 1e-6;
                close(g[k], (loss(a, &h, y) - loss(b, &h, y)) / 2e-6, 1e-8)
            }
        }),
        ("One source state is retrieved exactly", || {
            let (a, c) = attend([0.1, 0.2], &[[2.0, -3.0]]);
            assert_eq!(a, vec![1.0]);
            assert_eq!(c, [2.0, -3.0])
        }),
    ]
}
fn demo() {
    let h = states();
    for q in [[0.5, -0.25], [-1.0, 1.0]] {
        let (a, c) = attend(q, &h);
        println!(
            "query={q:?}\nweights={a:.6?}\ncontext={c:.6?}\nquery gradient={:.6?}",
            query_gradient(q, &h, [0.3, 0.8])
        );
    }
}
fn main() {
    println!("NN Chapter 09: Additive Attention and Query Gradients");
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
