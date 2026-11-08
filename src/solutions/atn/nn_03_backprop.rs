/*
NN Chapter 03. Differentiate and Train an MLP
[src/bin/nn_03_backprop.rs]

Goal:
Implement every gradient of a 2 -> 3 -> 1 tanh MLP, numerically check all
13 parameters, and learn XOR with full-batch gradient descent.

Why this matters for neural networks:
Chapter 2 supplied useful hidden features by hand. Backpropagation tells each
hidden connection how it contributed to an error; an optimizer uses those
derivatives to change the weights. Differentiation and optimization are separate.

Setup and dimensions:
x: 2, h: 3, output and target: scalar.
p[0..6]: three rows of two input weights; p[6..9]: hidden biases;
p[9..12]: output weights; p[12]: output bias.
This packed array is column-vector notation internally, documented by its indices.

Equations and gradient rules:
h_j = tanh(sum_i W1[j][i]*x_i + b1_j)
y_hat = sum_j W2[j]*h_j + b2
L = 0.5*(y_hat-y)^2; delta2 = y_hat-y
dW2_j = delta2*h_j; db2 = delta2
delta1_j = delta2*W2_j*(1-h_j^2)
dW1_ji = delta1_j*x_i; db1_j = delta1_j
batch gradient = sum_sample gradient / N
p <- p - lr*batch_gradient
finite difference = (L(p+epsilon e_j)-L(p-epsilon e_j))/(2 epsilon).

Implementation steps:
1. Cache hidden outputs from the forward pass.
2. Calculate all derivatives using the OLD weights.
3. Accumulate gradients over N samples and divide exactly once.
4. Update all 13 parameters, then measure loss with the new parameters.

Important limitations:
Use nonidentical initial weights to break hidden-unit symmetry. The half-squared
loss is summed over the one output and averaged over samples. This is a tiny
training-set fit, not held-out generalization. f64 improves finite-difference checks.

The jump to the next idea:
To process symbols or local patterns, first give inputs an appropriate representation.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_03_backprop
  cargo test --bin nn_03_backprop
Or, independently:
  rustc --edition=2021 nn_03_backprop.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://d2l.ai/chapter_multilayer-perceptrons/backprop.html
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

/// Hints: Cache each hidden tanh output. The output neuron is linear; do not apply tanh twice.
fn forward(p: &[f64], x: [f64; 2]) -> ([f64; 3], f64) {
    let mut h = [0.0; 3];
    for j in 0..3 {
        h[j] = (p[2 * j] * x[0] + p[2 * j + 1] * x[1] + p[6 + j]).tanh();
    }
    let out = (0..3).map(|j| p[9 + j] * h[j]).sum::<f64>() + p[12];
    (h, out)
}
/// Hints: Start at output error, then multiply by the OLD output weight and 1-h^2. Each input weight also receives its input coordinate.
fn gradients(p: &[f64], x: [f64; 2], y: f64) -> [f64; 13] {
    let (h, out) = forward(p, x);
    let error = out - y;
    let mut g = [0.0; 13];
    for j in 0..3 {
        let delta = error * p[9 + j] * (1.0 - h[j] * h[j]);
        g[2 * j] = delta * x[0];
        g[2 * j + 1] = delta * x[1];
        g[6 + j] = delta;
        g[9 + j] = error * h[j];
    }
    g[12] = error;
    g
}

fn initial() -> Vec<f64> {
    (0..13).map(|i| 0.4 * ((i + 1) as f64).sin()).collect()
}
fn dataset() -> Vec<([f64; 2], f64)> {
    vec![
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
}
fn loss(p: &[f64], x: [f64; 2], y: f64) -> f64 {
    0.5 * (forward(p, x).1 - y).powi(2)
}
fn mean_loss(p: &[f64], data: &[([f64; 2], f64)]) -> f64 {
    assert!(!data.is_empty());
    data.iter().map(|&(x, y)| loss(p, x, y)).sum::<f64>() / data.len() as f64
}
/// Hints: reset sums each epoch; never update p inside the sample loop.
/// Averaging is over samples, not over parameters or hidden coordinates.
fn train_mlp(data: &[([f64; 2], f64)], lr: f64, epochs: usize) -> Vec<f64> {
    assert!(!data.is_empty() && lr > 0.0 && lr.is_finite());
    let mut p = initial();
    for _ in 0..epochs {
        let mut sums = [0.0; 13];
        for &(x, y) in data {
            let g = gradients(&p, x, y);
            for j in 0..13 {
                sums[j] += g[j]
            }
        }
        for j in 0..13 {
            p[j] -= lr * sums[j] / data.len() as f64
        }
    }
    p
}

fn cases() -> Vec<Case> {
    vec![
        (
            "All 13 parameter gradients match finite differences",
            || {
                let p = initial();
                let x = [0.7, -0.4];
                let y = 0.3;
                let g = gradients(&p, x, y);
                let eps = 1e-6;
                for j in 0..13 {
                    let (mut a, mut b) = (p.clone(), p.clone());
                    a[j] += eps;
                    b[j] -= eps;
                    close(g[j], (loss(&a, x, y) - loss(&b, x, y)) / (2.0 * eps), 1e-8);
                }
            },
        ),
        ("Zero error gives zero gradients", || {
            let p = initial();
            let x = [0.2, 0.8];
            for g in gradients(&p, x, forward(&p, x).1) {
                close(g, 0.0, 1e-12)
            }
        }),
        ("One small step lowers mean loss", || {
            let d = dataset();
            assert!(mean_loss(&train_mlp(&d, 0.01, 1), &d) < mean_loss(&initial(), &d))
        }),
        ("Duplicate batch does not double averaged update", || {
            let d = dataset();
            let doubled = [d.clone(), d.clone()].concat();
            for (a, b) in train_mlp(&d, 0.1, 1)
                .iter()
                .zip(train_mlp(&doubled, 0.1, 1))
            {
                close(*a, b, 1e-12)
            }
        }),
        ("Training learns every XOR row", || {
            let d = dataset();
            let p = train_mlp(&d, 0.2, 20000);
            for (x, y) in d {
                close(forward(&p, x).1, y, 0.05)
            }
        }),
        ("Zero epochs preserves initialization", || {
            assert_eq!(train_mlp(&dataset(), 0.1, 0), initial())
        }),
    ]
}
fn demo() {
    let d = dataset();
    let p = train_mlp(&d, 0.2, 20000);
    println!(
        "Mean half-squared loss: {:.8} -> {:.8}",
        mean_loss(&initial(), &d),
        mean_loss(&p, &d)
    );
    for (x, y) in d {
        println!("x={x:?}, target={y}, prediction={:.6}", forward(&p, x).1);
    }
}
fn main() {
    println!("NN Chapter 03: Differentiate and Train an MLP");
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
