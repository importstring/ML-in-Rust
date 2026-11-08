/*
NN Chapter 08. Encoder-Decoder Flow and Teacher Forcing
[src/bin/nn_08_seq2seq.rs]

Goal:
Encode a variable-length source, decode a separately sized target, calculate
sequence likelihood, and contrast teacher forcing with autoregressive feedback.

Why this matters for neural networks:
Recurrence alone does not specify which outputs correspond to which inputs.
An encoder-decoder reads the source first, then conditions a new recurrence on
its context. Training can provide the correct previous token; generation cannot.

Setup and dimensions:
Source: T scalar features. Encoder context and decoder state: scalars.
Vocabulary size 4: BOS=0, A=1, B=2, EOS=3.
Targets: U token IDs, conventionally ending in EOS. Decoder output: U x 4 logits.
The first decoder input is BOS; subsequent inputs depend on the feedback mode.

Equations and gradient rules:
c_t=tanh(0.7*x_t+0.5*c_(t-1)); context=c_T
s_u=tanh(0.6*embedding(previous_token)+0.4*s_(u-1)+context)
logits_u[j]=s_u*output_weight[j]+output_bias[j]
P(y|x)=product_u P(y_u | y_<u, context)
mean NLL = -(1/U)*sum_u log P(y_u)
teacher-forcing input[u] = BOS if u=0 else target[u-1]
autoregressive input[u] = BOS if u=0 else prediction[u-1]
NLL(logits,y)=log(sum_j exp(logits[j]-m))+m-logits[y],
where m=max(logits), for stable calculation.

Implementation steps:
1. Finish encoding before initializing the decoder.
2. Record the token actually consumed at every step.
3. Predict before replacing previous_token with the target or prediction.
4. Generation stops on EOS or a strict length cap; targets are unavailable.

Important limitations:
Parameters are fixed. This exercise demonstrates conditional data flow and the
loss, NOT a trained translator. A greedy generated sequence is not necessarily
the globally most likely sequence. No attention is used yet.

The jump to the next idea:
Rather than passing only one context, attention lets each decoder query retrieve
a different weighted combination of all encoder states.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_08_seq2seq
  cargo test --bin nn_08_seq2seq
Or, independently:
  rustc --edition=2021 nn_08_seq2seq.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://arxiv.org/abs/1409.3215
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

const EMB: [f64; 4] = [0.0, 0.5, -0.5, 0.2];
fn encode(xs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &x in xs {
        h = (0.7 * x + 0.5 * h).tanh()
    }
    h
}
fn logits(context: f64, state: f64, previous: usize, bias: [f64; 4]) -> (f64, [f64; 4]) {
    assert!(previous < 4);
    let s = (0.6 * EMB[previous] + 0.4 * state + context).tanh();
    (
        s,
        [
            -0.3 * s + bias[0],
            s + bias[1],
            -s + bias[2],
            0.2 * s + bias[3],
        ],
    )
}
fn nll(z: &[f64; 4], y: usize) -> f64 {
    assert!(y < 4);
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    z.iter().map(|v| (v - m).exp()).sum::<f64>().ln() + m - z[y]
}
fn argmax(z: &[f64; 4]) -> usize {
    let mut best = 0;
    for j in 1..4 {
        if z[j] > z[best] {
            best = j
        }
    }
    best
}
/// Hints: score against the current target; feed the PREVIOUS target.
/// Both modes see the same BOS at step zero, so their first logits must match.
fn decode(c: f64, targets: &[usize], teacher: bool) -> (Vec<usize>, Vec<[f64; 4]>, f64) {
    assert!(!targets.is_empty());
    let (mut state, mut previous) = (0.0, 0);
    let (mut consumed, mut outputs, mut loss) = (Vec::new(), Vec::new(), 0.0);
    for &y in targets {
        consumed.push(previous);
        let (s, z) = logits(c, state, previous, [0.0; 4]);
        state = s;
        loss += nll(&z, y);
        previous = if teacher { y } else { argmax(&z) };
        outputs.push(z);
    }
    (consumed, outputs, loss / targets.len() as f64)
}
fn generate(c: f64, bias: [f64; 4], max_tokens: usize) -> Vec<usize> {
    let (mut state, mut previous) = (0.0, 0);
    let mut out = Vec::new();
    for _ in 0..max_tokens {
        let (s, z) = logits(c, state, previous, bias);
        state = s;
        previous = argmax(&z);
        out.push(previous);
        if previous == 3 {
            break;
        }
    }
    out
}

fn cases() -> Vec<Case> {
    vec![
        ("Teacher-forcing inputs are shifted targets", || {
            assert_eq!(decode(0.2, &[2, 1, 3], true).0, vec![0, 2, 1])
        }),
        ("Free-running inputs use previous predictions", || {
            let (inputs, z, _) = decode(0.2, &[2, 1, 3], false);
            assert_eq!(inputs[0], 0);
            for t in 1..3 {
                assert_eq!(inputs[t], argmax(&z[t - 1]))
            }
        }),
        ("Both modes have identical first-step logits", || {
            assert_eq!(
                decode(0.2, &[2, 1, 3], true).1[0],
                decode(0.2, &[2, 1, 3], false).1[0]
            )
        }),
        ("Source and output lengths may differ", || {
            assert_eq!(decode(encode(&[0.5, -0.1]), &[1, 2, 1, 3], true).1.len(), 4)
        }),
        ("Uniform four-token NLL is ln(4)", || {
            close(nll(&[0.0; 4], 2), 4_f64.ln(), 1e-12)
        }),
        ("Generation stops at EOS", || {
            assert_eq!(generate(0.2, [0.0, 0.0, 0.0, 10.0], 8), vec![3])
        }),
        ("Length cap prevents non-EOS loops", || {
            assert_eq!(generate(0.2, [0.0, 10.0, 0.0, -10.0], 5).len(), 5)
        }),
    ]
}
fn demo() {
    let c = encode(&[1.0, 0.0, -0.5]);
    println!("Encoder context={c:.6}");
    for teacher in [true, false] {
        let (inputs, z, loss) = decode(c, &[2, 1, 3], teacher);
        println!("teacher_forcing={teacher}, consumed={inputs:?}, mean_NLL={loss:.6}");
        for (t, row) in z.iter().enumerate() {
            println!(
                "step {t}: logits={row:.4?}, greedy prediction={}",
                argmax(row)
            );
        }
    }
    println!("Fixed weights: this is not a trained translator.");
}
fn main() {
    println!("NN Chapter 08: Encoder-Decoder Flow and Teacher Forcing");
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
