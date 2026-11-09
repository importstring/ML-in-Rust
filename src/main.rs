/*
NN Chapter 01. Train a Perceptron
[src/bin/nn_01_neurons.rs]

Goal:
Train one threshold neuron on AND, OR and NAND.
Then demonstrate why a single affine decision boundary cannot solve XOR.

Why this matters for neural networks:
A neuron is a dot product, bias and activation, not an entire learning system.
The update rule must be specified separately. The perceptron corrects classification
mistakes; it does NOT differentiate through the discontinuous threshold.

Setup and dimensions:
x: [f64; 2], w: [f64; 2], b: f64, target: 0 or 1.
One sample has two independent binary inputs. A dataset contains all four pairs.
Prediction uses >= 0, so the zero-initialized model initially predicts 1.

Equations and gradient rules:
z = w[0]*x[0] + w[1]*x[1] + b
y_hat = 1 if z >= 0, else 0
error = y - y_hat
w[i] <- w[i] + learning_rate * error * x[i]
b    <- b    + learning_rate * error
This is an online mistake-driven update, not MSE gradient descent.

XOR impossibility with this threshold convention:
negative (0,0): b < 0; negative (1,1): w0+w1+b < 0.
positive (1,0),(0,1): w0+b >= 0 and w1+b >= 0.
Adding positives gives w0+w1+2b >= 0, incompatible with negatives.

Implementation steps:
1. Implement predict without modifying weights.
2. Sweep the truth table, correcting each mistake immediately.
3. Stop only after an entire epoch has no mistakes, or the cap is reached.
4. Test all four inputs, not merely the examples printed in the demo.

Important limitations:
A finite epoch cap prevents hanging on nonseparable data. Non-convergence on XOR
is an expected limitation, NOT a failing correctness test. Training-set success
on four points is not evidence of generalization to arbitrary real inputs.

The jump to the next idea:
A nonlinear hidden representation can make XOR separable. Chapter 2 builds one.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_01_neurons
  cargo test --bin nn_01_neurons
Or, independently:
  rustc --edition=2021 nn_01_neurons.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://arxiv.org/html/1404.7828
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

#[derive(Debug, Clone)]
struct Perceptron {
    w: [f64; 2],
    b: f64,
}
impl Perceptron {
    /// Hints: multiply corresponding coordinates; add bias once; threshold last.
    fn predict(&self, x: [f64; 2]) -> i32 {
        (self.w[0] * x[0] + self.w[1] * x[1] + self.b >= 0.0) as i32
    }
}
/// Hints: label minus prediction sets the update sign; use the current weights.
/// An error-free epoch is the stopping condition, not just a small parameter change.
fn train(data: &[([f64; 2], i32)], lr: f64, epochs: usize) -> (Perceptron, bool) {
    assert!(!data.is_empty() && lr.is_finite() && lr > 0.0);
    assert!(data
        .iter()
        .all(|(x, y)| x.iter().all(|v| v.is_finite()) && (0..=1).contains(y)));
    let mut m = Perceptron {
        w: [0.0; 2],
        b: 0.0,
    };
    for _ in 0..epochs {
        let mut mistakes = 0;
        for &(x, y) in data {
            let error = y - m.predict(x);
            for i in 0..2 {
                m.w[i] += lr * error as f64 * x[i];
            }
            m.b += lr * error as f64;
            mistakes += usize::from(error != 0);
        }
        if mistakes == 0 {
            return (m, true);
        }
    }
    (m, false)
}
fn table(labels: [i32; 4]) -> Vec<([f64; 2], i32)> {
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        .into_iter()
        .zip(labels)
        .collect()
}
fn fit_table(labels: [i32; 4]) {
    let data = table(labels);
    let (m, done) = train(&data, 0.2, 100);
    assert!(done);
    for (x, y) in data {
        assert_eq!(m.predict(x), y);
    }
}

fn cases() -> Vec<Case> {
    vec![
        ("AND truth table", || fit_table([0, 0, 0, 1])),
        ("OR truth table", || fit_table([0, 1, 1, 1])),
        ("NAND truth table", || fit_table([1, 1, 1, 0])),
        ("Threshold boundary includes zero", || {
            assert_eq!(
                Perceptron {
                    w: [0.0; 2],
                    b: 0.0
                }
                    .predict([3.0, -2.0]),
                1
            )
        }),
        ("One negative example updates bias downward", || {
            let (m, _) = train(&[([0.0, 0.0], 0)], 0.2, 1);
            close(m.b, -0.2, 1e-12);
        }),
        ("XOR remains unsolved: expected limitation", || {
            let d = table([0, 1, 1, 0]);
            let (m, done) = train(&d, 0.2, 100);
            assert!(!done);
            assert!(d.iter().any(|&(x, y)| m.predict(x) != y));
        }),
    ]
}
fn demo() {
    let data = table([0, 0, 0, 1]);
    let (m, _) = train(&data, 0.2, 100);
    println!("Learned AND parameters: {m:?}");
    for (x, y) in data {
        println!("x={x:?}, target={y}, prediction={}", m.predict(x));
    }
}
fn main() {
    println!("NN Chapter 01: Train a Perceptron");
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
