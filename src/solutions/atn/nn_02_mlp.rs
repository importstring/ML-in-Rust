/*
NN Chapter 02. Build a Nonlinear Multilayer Perceptron
[src/bin/nn_02_mlp.rs]

Goal:
Build reusable dense layers, compose them with ReLU, and solve XOR by construction.

Why this matters for neural networks:
A collection of affine layers without an intervening nonlinearity collapses to
another affine map. Depth becomes useful here because ReLU changes the coordinates
in which the next layer makes its decision.

Setup and dimensions:
Row-vector convention, matching the supplied dense-layer exercise:
x: D; W: D x O; b: O; forward(x): O.
The XOR model uses a 2 -> 2 ReLU layer and a 2 -> 1 linear output.
The website sometimes uses column vectors; transposing W reconciles conventions.

Equations and gradient rules:
dense(x)[j] = sum_i x[i]*W[i][j] + b[j]
ReLU(z) = max(0,z)
h = ReLU(x W1 + b1); output = h W2 + b2
h[0] = max(0,x0-x1); h[1] = max(0,x1-x0)
output = h[0]+h[1] = |x0-x1| for binary inputs
Without ReLU: (x W1+b1) W2+b2 = x(W1 W2)+(b1 W2+b2).

Implementation steps:
1. Validate rectangular D x O weights, D inputs and O biases.
2. Sum along D for each output; add its bias exactly once.
3. Apply ReLU elementwise to hidden activations only.
4. Test XOR, a nonsquare dense matrix, and the affine-collapse counterexample.

Important limitations:
These weights are hand-designed, not learned. Matching XOR does not establish a
training algorithm. Zero-width and mismatched shapes are rejected by assertions.

The jump to the next idea:
Chapter 3 learns hidden weights rather than prescribing them, using backpropagation.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_02_mlp
  cargo test --bin nn_02_mlp
Or, independently:
  rustc --edition=2021 nn_02_mlp.rs -o lesson
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

#[derive(Clone)]
struct Dense {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}
impl Dense {
    /// Hints: outer index j selects an output; inner index i contracts inputs.
    /// Vec<Vec<_>> does not enforce a rectangular matrix: check it explicitly.
    fn forward(&self, x: &[f64]) -> Vec<f64> {
        assert!(!self.w.is_empty() && !self.b.is_empty());
        assert_eq!(x.len(), self.w.len());
        assert!(self.w.iter().all(|row| row.len() == self.b.len()));
        (0..self.b.len())
            .map(|j| self.b[j] + (0..x.len()).map(|i| x[i] * self.w[i][j]).sum::<f64>())
            .collect()
    }
}
/// Hints: preserve vector length; negative values become zero; zero stays zero.
fn relu(xs: Vec<f64>) -> Vec<f64> {
    xs.into_iter().map(|x| x.max(0.0)).collect()
}
fn xor(x: &[f64], nonlinear: bool) -> f64 {
    let first = Dense {
        w: vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
        b: vec![0.0; 2],
    };
    let second = Dense {
        w: vec![vec![1.0], vec![1.0]],
        b: vec![0.0],
    };
    let h = first.forward(x);
    second.forward(&if nonlinear { relu(h) } else { h })[0]
}

fn cases() -> Vec<Case> {
    vec![
        ("All XOR inputs", || {
            for (x, y) in [
                ([0.0, 0.0], 0.0),
                ([0.0, 1.0], 1.0),
                ([1.0, 0.0], 1.0),
                ([1.0, 1.0], 0.0),
            ] {
                close(xor(&x, true), y, 1e-12)
            }
        }),
        ("Nonsquare 2-to-3 dense layer", || {
            let m = Dense {
                w: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                b: vec![0.1, 0.2, 0.3],
            };
            let y = m.forward(&[2.0, -1.0]);
            for (a, b) in y.iter().zip([-1.9, -0.8, 0.3]) {
                close(*a, b, 1e-12)
            }
        }),
        ("Zero input returns biases", || {
            assert_eq!(
                Dense {
                    w: vec![vec![4.0, -2.0]],
                    b: vec![0.5, 1.5]
                }
                .forward(&[0.0]),
                vec![0.5, 1.5]
            )
        }),
        ("ReLU boundary and sign", || {
            assert_eq!(relu(vec![-2.0, 0.0, 3.0]), vec![0.0, 0.0, 3.0])
        }),
        ("Removing ReLU destroys this XOR representation", || {
            close(xor(&[0.0, 1.0], false), 0.0, 1e-12)
        }),
        ("Affine composition agrees with collapsed map", || {
            let a = Dense {
                w: vec![vec![2.0, 3.0]],
                b: vec![1.0, -1.0],
            };
            let b = Dense {
                w: vec![vec![4.0], vec![5.0]],
                b: vec![2.0],
            };
            for x in [-2.0, 0.0, 1.5] {
                close(b.forward(&a.forward(&[x]))[0], 23.0 * x + 1.0, 1e-12)
            }
        }),
    ]
}
fn demo() {
    for x in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
        println!(
            "x={x:?}: with ReLU={}, without ReLU={}",
            xor(&x, true),
            xor(&x, false)
        );
    }
}
fn main() {
    println!("NN Chapter 02: Build a Nonlinear Multilayer Perceptron");
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
