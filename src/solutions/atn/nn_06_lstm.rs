/*
NN Chapter 06. Build an LSTM Cell and Its Complete Backward Pass
[src/bin/nn_06_lstm.rs]

Goal:
Implement all four gate computations, both states, local backward derivatives,
full sequence BPTT, and a small supervised training loop.

Why this matters for neural networks:
The RNN repeatedly transforms its entire state. An LSTM also maintains a cell
updated by an additive path. The gates are computed functions with parameters;
they are not independent constants during the full backward pass.

Setup and dimensions:
To make every derivative inspectable, D=H=1. Each gate has three parameters:
input weight, previous-hidden weight, bias. p is [f,i,g,o], three entries per gate.
Cache x, previous h and c, gates f/i/g/o, new c and new h.
For H hidden units and D inputs the general count is 4H(D+H+1); here it is 12.

Equations and gradient rules:
a_k = p[3k]*x + p[3k+1]*h_prev + p[3k+2]
f=sigmoid(a_f); i=sigmoid(a_i); g=tanh(a_g); o=sigmoid(a_o)
c=f*c_prev+i*g; h=o*tanh(c)
q = incoming_dc + incoming_dh*o*(1-tanh(c)^2)
df_pre=q*c_prev*f*(1-f); di_pre=q*g*i*(1-i)
dg_pre=q*i*(1-g^2); do_pre=incoming_dh*tanh(c)*o*(1-o)
dp_k = gate_delta_k * [x,h_prev,1]
dx=sum_k gate_delta_k*input_weight_k
dh_prev=sum_k gate_delta_k*hidden_weight_k; dc_prev=q*f
The direct cell edge has derivative f; total recurrence also includes hidden/gate paths.

Implementation steps:
1. Compute ALL gates from the old hidden state.
2. Update c, then use the new c to compute h.
3. Combine incoming cell and hidden gradients into q.
4. Differentiate each gate and accumulate parameter contributions across time.
5. Pass BOTH dh_prev and dc_prev to the preceding cached step.

Important limitations:
The scalar model is a minimal exact instance, not a vector LSTM library.
Numerical checks cover all 12 parameters plus x, h_prev and c_prev.
Gate values 0 and 1 in the limiting-case tests are manually supplied boundary
values; finite sigmoid logits do not attain those exact endpoints.

The jump to the next idea:
GRU removes the separate cell state while preserving a gated additive update.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_06_lstm
  cargo test --bin nn_06_lstm
Or, independently:
  rustc --edition=2021 nn_06_lstm.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://d2l.ai/chapter_recurrent-modern/lstm.html
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
#[derive(Clone, Copy)]
struct Cache {
    x: f64,
    hp: f64,
    cp: f64,
    f: f64,
    i: f64,
    g: f64,
    o: f64,
    c: f64,
    h: f64,
}
/// Hints: preserve old c until its contribution is used; h exposes the NEW c.
fn update_state(cp: f64, f: f64, i: f64, g: f64, o: f64) -> (f64, f64) {
    let c = f * cp + i * g;
    (c, o * c.tanh())
}
/// Hints: use the previous h in every gate; candidate uses tanh, not sigmoid.
fn step(p: &[f64; 12], x: f64, hp: f64, cp: f64) -> Cache {
    let a = |k: usize| p[3 * k] * x + p[3 * k + 1] * hp + p[3 * k + 2];
    let (f, i, g, o) = (sigmoid(a(0)), sigmoid(a(1)), a(2).tanh(), sigmoid(a(3)));
    let (c, h) = update_state(cp, f, i, g, o);
    Cache {
        x,
        hp,
        cp,
        f,
        i,
        g,
        o,
        c,
        h,
    }
}
/// Hints: q joins two routes into c; the sigmoid derivative is gate*(1-gate).
/// Return both recurrent gradients. Omitting dh_prev loses the gate feedback path.
fn backward(p: &[f64; 12], s: Cache, dh: f64, dc: f64) -> ([f64; 12], f64, f64, f64) {
    let q = dc + dh * s.o * (1.0 - s.c.tanh().powi(2));
    let da = [
        q * s.cp * s.f * (1.0 - s.f),
        q * s.g * s.i * (1.0 - s.i),
        q * s.i * (1.0 - s.g * s.g),
        dh * s.c.tanh() * s.o * (1.0 - s.o),
    ];
    let (mut dp, mut dx, mut dhp) = ([0.0; 12], 0.0, 0.0);
    for k in 0..4 {
        dp[3 * k] = da[k] * s.x;
        dp[3 * k + 1] = da[k] * s.hp;
        dp[3 * k + 2] = da[k];
        dx += da[k] * p[3 * k];
        dhp += da[k] * p[3 * k + 1];
    }
    (dp, dx, dhp, q * s.f)
}
fn initial() -> [f64; 12] {
    [0.2, 0.1, 1.0, 0.3, -0.2, 0.0, 0.8, 0.3, 0.0, 0.2, 0.1, 0.0]
}
fn sequence(p: &[f64; 12], xs: &[f64]) -> Vec<Cache> {
    let (mut h, mut c) = (0.0, 0.0);
    let mut states = Vec::new();
    for &x in xs {
        let s = step(p, x, h, c);
        h = s.h;
        c = s.c;
        states.push(s)
    }
    states
}
fn loss(p: &[f64; 12], xs: &[f64], y: f64) -> f64 {
    0.5 * (sequence(p, xs).last().expect("nonempty sequence").h - y).powi(2)
}
fn sequence_gradient(p: &[f64; 12], xs: &[f64], y: f64) -> [f64; 12] {
    let states = sequence(p, xs);
    let mut dh = states.last().expect("nonempty sequence").h - y;
    let mut dc = 0.0;
    let mut sum = [0.0; 12];
    for &s in states.iter().rev() {
        let (g, _, hp, cp) = backward(p, s, dh, dc);
        for j in 0..12 {
            sum[j] += g[j]
        }
        dh = hp;
        dc = cp;
    }
    sum
}
fn train(epochs: usize) -> [f64; 12] {
    let mut p = initial();
    for _ in 0..epochs {
        let g = sequence_gradient(&p, &[0.4, -0.1, 0.2], 0.35);
        for j in 0..12 {
            p[j] -= 0.5 * g[j]
        }
    }
    p
}

fn cases() -> Vec<Case> {
    vec![
        ("Worked cell c=0.98 and h=0.602453", || {
            let (c, h) = update_state(1.2, 0.9, 0.2, -0.5, 0.8);
            close(c, 0.98, 1e-12);
            close(h, 0.602453, 1e-6)
        }),
        ("Closing output gate preserves cell but hides it", || {
            let (c, h) = update_state(1.2, 1.0, 0.0, 0.5, 0.0);
            close(h, 0.0, 0.0);
            close(c, 1.2, 0.0)
        }),
        ("All 12 local parameter derivatives", || {
            let p = initial();
            let s = step(&p, 0.3, 0.2, -0.1);
            let (g, _, _, _) = backward(&p, s, 0.7, -0.4);
            let f = |p: &[f64; 12]| {
                let s = step(p, 0.3, 0.2, -0.1);
                0.7 * s.h - 0.4 * s.c
            };
            for j in 0..12 {
                let (mut a, mut b) = (p, p);
                a[j] += 1e-6;
                b[j] -= 1e-6;
                close(g[j], (f(&a) - f(&b)) / 2e-6, 1e-8)
            }
        }),
        ("Input and both previous-state derivatives", || {
            let p = initial();
            let s = step(&p, 0.3, 0.2, -0.1);
            let (_, dx, dh, dc) = backward(&p, s, 0.7, -0.4);
            let f = |v: [f64; 3]| {
                let s = step(&p, v[0], v[1], v[2]);
                0.7 * s.h - 0.4 * s.c
            };
            for (j, g) in [dx, dh, dc].iter().enumerate() {
                let (mut a, mut b) = ([0.3, 0.2, -0.1], [0.3, 0.2, -0.1]);
                a[j] += 1e-6;
                b[j] -= 1e-6;
                close(*g, (f(a) - f(b)) / 2e-6, 1e-8)
            }
        }),
        ("All shared BPTT parameter derivatives", || {
            let p = initial();
            let x = [0.4, -0.1, 0.2];
            let g = sequence_gradient(&p, &x, 0.35);
            for j in 0..12 {
                let (mut a, mut b) = (p, p);
                a[j] += 1e-6;
                b[j] -= 1e-6;
                close(g[j], (loss(&a, &x, 0.35) - loss(&b, &x, 0.35)) / 2e-6, 1e-8)
            }
        }),
        ("Sequence training lowers supervised loss", || {
            assert!(loss(&train(2000), &[0.4, -0.1, 0.2], 0.35) < 1e-5)
        }),
        ("Direct-path retention calculation", || {
            close(0.99_f64.powi(100), 0.3660323412732292, 1e-12)
        }),
    ]
}
fn demo() {
    let p = initial();
    for (t, s) in sequence(&p, &[0.4, -0.1, 0.2]).iter().enumerate() {
        println!(
            "t={} f={:.5} i={:.5} g={:.5} o={:.5} c={:.5} h={:.5}",
            t + 1,
            s.f,
            s.i,
            s.g,
            s.o,
            s.c,
            s.h
        );
    }
    let trained = train(2000);
    println!(
        "One-sequence training loss {:.8} -> {:.8}",
        loss(&p, &[0.4, -0.1, 0.2], 0.35),
        loss(&trained, &[0.4, -0.1, 0.2], 0.35)
    );
    println!("This single-sequence fit is not a generalization claim.");
}
fn main() {
    println!("NN Chapter 06: Build an LSTM Cell and Its Complete Backward Pass");
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
