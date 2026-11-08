/*
NN Chapter 05. Unroll an RNN and Backpropagate Through Time
[src/bin/nn_05_rnn.rs]

Goal:
Implement a scalar recurrent network, differentiate a final-state loss through
every step, and train its three shared parameters on a small sequence dataset.

Why this matters for neural networks:
An MLP consumes a fixed representation. An RNN reuses a transition and carries
state between positions. Reusing parameters means gradient contributions must
be added across time, not applied as independent per-position weights.

Setup and dimensions:
Sequence x: T scalars. Hidden h[t]: scalar, with h[0]=initial state.
Parameters p=[input_weight, recurrent_weight, bias].
Only the final state has a supervised target in this exercise.

Equations and gradient rules:
a_t = p0*x_t + p1*h_(t-1) + p2; h_t=tanh(a_t)
L=0.5*(h_T-y)^2; incoming dh_T=h_T-y
da_t = dh_t*(1-h_t^2)
dp0 += da_t*x_t; dp1 += da_t*h_(t-1); dp2 += da_t
dh_(t-1) = da_t*p1
dh_T/dh_0 = product_t p1*(1-h_t^2)
The power examples 0.5^20 and 1.5^20 illustrate repeated factors;
they are NOT guaranteed actual RNN Jacobians.

Implementation steps:
1. Save h0 and every subsequent state; there are T+1 cached states.
2. Walk backwards with the final loss derivative.
3. Accumulate three parameter gradients, then update once per batch.
4. Compare analytic gradients and the initial-state derivative with finite differences.

Important limitations:
This is full BPTT, not truncated BPTT. No clipping is applied, so the tests choose
small bounded inputs. The toy fit measures training loss only. The hidden state
is bounded by tanh, but products of recurrent Jacobians can still be unstable.

The jump to the next idea:
LSTM introduces a separate additive cell path and gates to control information flow.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_05_rnn
  cargo test --bin nn_05_rnn
Or, independently:
  rustc --edition=2021 nn_05_rnn.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://d2l.ai/chapter_recurrent-neural-networks/bptt.html
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

fn unroll(p: &[f64; 3], xs: &[f64], h0: f64) -> Vec<f64> {
    let mut states = vec![h0];
    for &x in xs {
        let old = *states.last().unwrap();
        states.push((p[0] * x + p[1] * old + p[2]).tanh());
    }
    states
}
fn loss(p: &[f64; 3], xs: &[f64], y: f64, h0: f64) -> f64 {
    0.5 * (unroll(p, xs, h0).last().unwrap() - y).powi(2)
}
/// Hints: states[t] is the PREVIOUS state for input xs[t].
/// The incoming gradient is multiplied by tanh's derivative exactly once.
fn bptt(p: &[f64; 3], xs: &[f64], y: f64, h0: f64) -> ([f64; 3], f64) {
    let h = unroll(p, xs, h0);
    let mut dh = h.last().unwrap() - y;
    let mut g = [0.0; 3];
    for t in (0..xs.len()).rev() {
        let da = dh * (1.0 - h[t + 1] * h[t + 1]);
        g[0] += da * xs[t];
        g[1] += da * h[t];
        g[2] += da;
        dh = da * p[1];
    }
    (g, dh)
}
fn data() -> Vec<(Vec<f64>, f64)> {
    let mut d = Vec::new();
    for a in [-0.2, 0.0, 0.2] {
        for b in [-0.2, 0.0, 0.2] {
            d.push((vec![a, b], a + b))
        }
    }
    d
}
fn mean_loss(p: &[f64; 3], d: &[(Vec<f64>, f64)]) -> f64 {
    d.iter().map(|(x, y)| loss(p, x, *y, 0.0)).sum::<f64>() / d.len() as f64
}
fn train(d: &[(Vec<f64>, f64)], epochs: usize) -> [f64; 3] {
    assert!(!d.is_empty());
    let mut p = [0.5, 0.3, 0.0];
    for _ in 0..epochs {
        let mut sum = [0.0; 3];
        for (x, y) in d {
            let (g, _) = bptt(&p, x, *y, 0.0);
            for j in 0..3 {
                sum[j] += g[j]
            }
        }
        for j in 0..3 {
            p[j] -= 0.3 * sum[j] / d.len() as f64;
        }
    }
    p
}

fn cases() -> Vec<Case> {
    vec![
        ("Unroll caches T+1 states", || {
            assert_eq!(unroll(&[1.0, 0.5, 0.0], &[1.0, 0.0, 0.0], 0.0).len(), 4)
        }),
        ("All shared-parameter gradients", || {
            let p = [0.8, 0.6, 0.1];
            let x = [0.3, -0.2, 0.5];
            let (g, _) = bptt(&p, &x, 0.2, 0.1);
            for j in 0..3 {
                let (mut a, mut b) = (p, p);
                a[j] += 1e-6;
                b[j] -= 1e-6;
                close(
                    g[j],
                    (loss(&a, &x, 0.2, 0.1) - loss(&b, &x, 0.2, 0.1)) / 2e-6,
                    1e-8,
                )
            }
        }),
        ("Initial-state gradient includes every time step", || {
            let p = [0.8, 0.6, 0.1];
            let x = [0.3, -0.2, 0.5];
            let (_, g) = bptt(&p, &x, 0.2, 0.1);
            close(
                g,
                (loss(&p, &x, 0.2, 0.100001) - loss(&p, &x, 0.2, 0.099999)) / 2e-6,
                1e-8,
            );
        }),
        ("Zero recurrent weight removes old-state dependence", || {
            close(
                *unroll(&[1.0, 0.0, 0.0], &[0.8, -0.2], 0.9).last().unwrap(),
                (-0.2_f64).tanh(),
                1e-12,
            )
        }),
        ("Vanishing and exploding product examples", || {
            assert!(0.5_f64.powi(20) < 1e-6);
            assert!(1.5_f64.powi(20) > 3000.0)
        }),
        ("Training reduces sequence-sum loss", || {
            let d = data();
            assert!(mean_loss(&train(&d, 4000), &d) < 0.02 * mean_loss(&[0.5, 0.3, 0.0], &d));
        }),
    ]
}
fn demo() {
    let d = data();
    let p = train(&d, 4000);
    println!(
        "Parameters={p:?}, training mean loss={:.8}",
        mean_loss(&p, &d)
    );
    for (x, y) in d {
        println!(
            "sequence={x:?}, target={y:.3}, prediction={:.4}",
            unroll(&p, &x, 0.0).last().unwrap()
        );
    }
}
fn main() {
    println!("NN Chapter 05: Unroll an RNN and Backpropagate Through Time");
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
