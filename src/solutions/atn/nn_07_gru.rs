/*
NN Chapter 07. A GRU Cell with Reset and Update Gradients
[src/bin/nn_07_gru.rs]

Goal:
Implement a scalar GRU and differentiate its output through every gate,
including the reset gate's contribution to the candidate.

Why this matters for neural networks:
The LSTM keeps hidden and cell states. The GRU keeps one state and uses the
update gate to interpolate between the old state and a proposed replacement.
The reset gate controls how the old state contributes to that proposal.

Setup and dimensions:
x, h_prev, output: scalars. p has 9 entries: [z_x,z_h,z_b,
r_x,r_h,r_b,g_x,g_h,g_b]. There is no separate cell c.
Convention: z=1 RETAINS the old state. Some libraries reverse that convention.

Equations and gradient rules:
z=sigmoid(z_x*x+z_h*h_prev+z_b)
r=sigmoid(r_x*x+r_h*h_prev+r_b)
g=tanh(g_x*x+g_h*(r*h_prev)+g_b)
h=z*h_prev+(1-z)*g
Given incoming dh:
dz_pre=dh*(h_prev-g)*z*(1-z)
dg_pre=dh*(1-z)*(1-g^2)
dr_pre=dg_pre*g_h*h_prev*r*(1-r)
dh_prev=dh*z+dz_pre*z_h+dr_pre*r_h+dg_pre*g_h*r
dx=dz_pre*z_x+dr_pre*r_x+dg_pre*g_x
The direct retention edge contributes dh*z; it is not the whole derivative.

Implementation steps:
1. Compute z and r from the same old hidden state.
2. Multiply old h by r before applying the candidate recurrent weight.
3. Differentiate candidate before reset, since reset's derivative depends on it.
4. Check all 9 parameters, x and h_prev numerically.

Important limitations:
This is the scalar reset-before candidate convention. Vector reset-before and
reset-after formulations are not generally identical. It is a forward/backward
kernel exercise, not training a sequence task.

The jump to the next idea:
An encoder-decoder uses recurrence to map an input sequence to a separately
generated output sequence; input and output lengths need not match.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_07_gru
  cargo test --bin nn_07_gru
Or, independently:
  rustc --edition=2021 nn_07_gru.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://d2l.ai/chapter_recurrent-modern/gru.html
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

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn step(p: &[f64; 9], x: f64, h: f64) -> (f64, [f64; 3]) {
    let z = sigmoid(p[0] * x + p[1] * h + p[2]);
    let r = sigmoid(p[3] * x + p[4] * h + p[5]);
    let g = (p[6] * x + p[7] * (r * h) + p[8]).tanh();
    (z * h + (1.0 - z) * g, [z, r, g])
}
/// Hints: differentiate the interpolation first. The reset gate is reached
/// through g, so its derivative uses dg_pre, not the raw incoming dh.
fn backward(p: &[f64; 9], x: f64, h: f64, dh: f64) -> ([f64; 9], f64, f64) {
    let (_, [z, r, g]) = step(p, x, h);
    let dz = dh * (h - g) * z * (1.0 - z);
    let dg = dh * (1.0 - z) * (1.0 - g * g);
    let dr = dg * p[7] * h * r * (1.0 - r);
    let dp = [
        dz * x,
        dz * h,
        dz,
        dr * x,
        dr * h,
        dr,
        dg * x,
        dg * r * h,
        dg,
    ];
    (
        dp,
        dz * p[0] + dr * p[3] + dg * p[6],
        dh * z + dz * p[1] + dr * p[4] + dg * p[7] * r,
    )
}
fn initial() -> [f64; 9] {
    [0.4, 0.2, 1.0, -0.2, 0.3, 0.0, 0.8, 0.5, 0.0]
}
fn blend(h: f64, g: f64, z: f64) -> f64 {
    z * h + (1.0 - z) * g
}

fn cases() -> Vec<Case> {
    vec![
        ("Update gate one retains old state", || {
            close(blend(0.7, -0.2, 1.0), 0.7, 0.0)
        }),
        ("Update gate zero uses candidate", || {
            close(blend(0.7, -0.2, 0.0), -0.2, 0.0)
        }),
        ("Convex combination stays between endpoints", || {
            for z in [0.0, 0.25, 0.5, 1.0] {
                let h = blend(-0.5, 0.8, z);
                assert!((-0.5..=0.8).contains(&h))
            }
        }),
        ("All nine parameter derivatives", || {
            let p = initial();
            let (g, _, _) = backward(&p, 0.3, -0.2, 1.0);
            for j in 0..9 {
                let (mut a, mut b) = (p, p);
                a[j] += 1e-6;
                b[j] -= 1e-6;
                close(
                    g[j],
                    (step(&a, 0.3, -0.2).0 - step(&b, 0.3, -0.2).0) / 2e-6,
                    1e-8,
                )
            }
        }),
        ("Input and hidden derivatives", || {
            let p = initial();
            let (_, dx, dh) = backward(&p, 0.3, -0.2, 1.0);
            close(
                dx,
                (step(&p, 0.300001, -0.2).0 - step(&p, 0.299999, -0.2).0) / 2e-6,
                1e-8,
            );
            close(
                dh,
                (step(&p, 0.3, -0.199999).0 - step(&p, 0.3, -0.200001).0) / 2e-6,
                1e-8,
            );
        }),
        (
            "Zero old state makes reset-parameter gradients zero",
            || {
                let (g, _, _) = backward(&initial(), 0.3, 0.0, 1.0);
                for v in &g[3..6] {
                    close(*v, 0.0, 0.0)
                }
            },
        ),
    ]
}
fn demo() {
    let p = initial();
    let mut h = 0.0;
    for x in [1.0, 0.0, -0.5] {
        let (next, [z, r, g]) = step(&p, x, h);
        println!("x={x}, old_h={h:.6}, z={z:.6}, r={r:.6}, candidate={g:.6}, new_h={next:.6}");
        h = next;
    }
}
fn main() {
    println!("NN Chapter 07: A GRU Cell with Reset and Update Gradients");
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
