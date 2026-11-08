/*
NN Chapter 12. Train a Next-Token Model and Decode
[src/bin/nn_12_language_models.rs]

Goal:
Implement stable cross-entropy, its logit gradients, full-batch SGD, perplexity,
temperature scaling and EOS-aware generation. Also compute a masked-position loss.

Why this matters for neural networks:
A Transformer produces logits, but it still needs an objective and optimizer.
To isolate those mechanisms without hiding a Transformer backward pass, this
exercise trains a categorical bigram model: a learned next-token logit row per ID.

Setup and dimensions:
V=4: BOS=0, A=1, B=2, EOS=3. Parameters W: 4 x 4.
For previous token i, logits=W[i]; no embeddings, attention or hidden layers.
Dataset pairs are shifted sequence positions. W starts at zero.
The separate masked-loss example accepts per-position logits and selected indices.

Equations and gradient rules:
P(next=j|previous=i)=softmax(W[i])[j]
L_i=logsumexp(W[i])-W[i][target]
dW[i][j] += P(j|i)-1[j=target]; all other rows get no contribution
batch_loss=sum_pairs L/N; batch_gradient=sum_pairs gradient/N
W <- W-lr*batch_gradient
perplexity=exp(mean_NLL), with natural logarithms
temperature probabilities=softmax(logits/temperature), temperature>0
masked_loss=mean NLL only over SELECTED prediction positions.
Masked loss differs from next-token loss; it is not an autoregressive decoder.

Implementation steps:
1. Shift a token sequence into (previous,target) pairs without wraparound.
2. Implement logsumexp so large logits do not overflow.
3. Accumulate all gradient rows, average once, and update.
4. Decode by feeding each predicted token back as the next input.
5. Stop on EOS or the token limit. Report training loss, not fabricated test accuracy.

Important limitations:
This is a trained BIGRAM model, not a trained Transformer or BERT.
The masked-loss helper demonstrates the selected-position objective only; it
does not implement BERT corruption or bidirectional contextual representations.
Greedy decoding is deterministic; temperatures are shown as distributions, not
claimed stochastic sampling. No Adam, top-k sampling or KV cache is implemented.

The jump to the next idea:
To scale up, replace the bigram logits with a differentiable Transformer,
backpropagate through every block, and evaluate on genuinely held-out text.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_12_language_models
  cargo test --bin nn_12_language_models
Or, independently:
  rustc --edition=2021 nn_12_language_models.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://arxiv.org/html/1810.04805
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

type Row = [f64; 4];
type Model = [Row; 4];
fn probabilities(z: &Row, temp: f64) -> Row {
    assert!(temp.is_finite() && temp > 0.0);
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Subtract max BEFORE dividing: avoids overflow from a large common offset.
    let e = z.map(|v| ((v - m) / temp).exp());
    let sum: f64 = e.iter().sum();
    e.map(|v| v / sum)
}
fn nll(z: &Row, y: usize) -> f64 {
    assert!(y < 4);
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    z.iter().map(|v| (v - m).exp()).sum::<f64>().ln() + (m - z[y])
}
fn pairs(tokens: &[usize]) -> Vec<(usize, usize)> {
    assert!(tokens.iter().all(|&x| x < 4));
    tokens.windows(2).map(|w| (w[0], w[1])).collect()
}
fn loss(w: &Model, data: &[(usize, usize)]) -> f64 {
    assert!(!data.is_empty());
    data.iter().map(|&(x, y)| nll(&w[x], y)).sum::<f64>() / data.len() as f64
}
/// Hints: differentiate cross-entropy with respect to logits, not probabilities.
/// Only the row selected by the previous token receives this sample's gradient.
fn gradient(w: &Model, data: &[(usize, usize)]) -> Model {
    assert!(!data.is_empty());
    let mut g = [[0.0; 4]; 4];
    for &(x, y) in data {
        assert!(x < 4 && y < 4);
        let p = probabilities(&w[x], 1.0);
        for j in 0..4 {
            g[x][j] += p[j] - if j == y { 1.0 } else { 0.0 };
        }
    }
    for row in &mut g {
        for v in row {
            *v /= data.len() as f64
        }
    }
    g
}
fn train(data: &[(usize, usize)], epochs: usize) -> Model {
    let mut w = [[0.0; 4]; 4];
    for _ in 0..epochs {
        let g = gradient(&w, data);
        for i in 0..4 {
            for j in 0..4 {
                w[i][j] -= 0.8 * g[i][j]
            }
        }
    }
    w
}
fn greedy(z: &Row) -> usize {
    let mut best = 0;
    for j in 1..4 {
        if z[j] > z[best] {
            best = j
        }
    }
    best
}
fn generate(w: &Model, start: usize, limit: usize) -> Vec<usize> {
    assert!(start < 4);
    let mut out = vec![start];
    for _ in 0..limit {
        let previous = *out.last().unwrap();
        if previous == 3 {
            break;
        }
        let next = greedy(&w[previous]);
        out.push(next);
        if next == 3 {
            break;
        }
    }
    out
}
fn masked_loss(logits: &[Row], targets: &[usize], selected: &[usize]) -> f64 {
    assert_eq!(logits.len(), targets.len());
    assert!(!selected.is_empty());
    assert!(selected.iter().all(|&i| i < logits.len()));
    selected
        .iter()
        .map(|&i| nll(&logits[i], targets[i]))
        .sum::<f64>()
        / selected.len() as f64
}

fn cases() -> Vec<Case> {
    vec![
        ("Sequence is shifted without wrapping EOS to BOS", || {
            assert_eq!(pairs(&[0, 1, 2, 3]), vec![(0, 1), (1, 2), (2, 3)])
        }),
        ("Uniform NLL and perplexity", || {
            let l = nll(&[0.0; 4], 2);
            close(l, 4_f64.ln(), 1e-12);
            close(l.exp(), 4.0, 1e-12)
        }),
        (
            "Every model-logit derivative matches finite differences",
            || {
                let w = [[0.2, -0.1, 0.5, 0.0]; 4];
                let d = pairs(&[0, 1, 2, 3]);
                let g = gradient(&w, &d);
                for i in 0..4 {
                    for j in 0..4 {
                        let (mut a, mut b) = (w, w);
                        a[i][j] += 1e-6;
                        b[i][j] -= 1e-6;
                        close(g[i][j], (loss(&a, &d) - loss(&b, &d)) / 2e-6, 1e-8)
                    }
                }
            },
        ),
        ("Training lowers NLL and generates the learned path", || {
            let d = pairs(&[0, 1, 2, 3]);
            let w = train(&d, 1500);
            assert!(loss(&w, &d) < 0.01);
            assert_eq!(generate(&w, 0, 10), vec![0, 1, 2, 3]);
        }),
        (
            "Repeated targets learn an empirical conditional frequency",
            || {
                let w = train(&[(0, 1), (0, 1), (0, 1), (0, 2)], 5000);
                let p = probabilities(&w[0], 1.0);
                close(p[1], 0.75, 0.01);
                close(p[2], 0.25, 0.01);
            },
        ),
        (
            "Temperature flattens probabilities but preserves argmax",
            || {
                let z = [-1.0, 0.0, 2.0, 0.5];
                let cold = probabilities(&z, 0.5);
                let hot = probabilities(&z, 2.0);
                assert!(cold[2] > hot[2]);
                assert_eq!(greedy(&cold), greedy(&hot));
                close(hot.iter().sum(), 1.0, 1e-12);
            },
        ),
        ("Masked loss ignores unselected positions", || {
            let mut z = [[0.0; 4]; 3];
            let before = masked_loss(&z, &[1, 2, 3], &[1]);
            z[0] = [500.0, -500.0, 20.0, 30.0];
            close(masked_loss(&z, &[1, 2, 3], &[1]), before, 0.0);
        }),
        ("Large common logit shift preserves NLL", || {
            close(
                nll(&[1000.0, 1001.0, 1002.0, 1003.0], 2),
                nll(&[0.0, 1.0, 2.0, 3.0], 2),
                1e-12,
            )
        }),
    ]
}
fn demo() {
    let d = pairs(&[0, 1, 2, 3]);
    let w = train(&d, 1500);
    let l = loss(&w, &d);
    println!("Training NLL={l:.6}, training perplexity={:.6}", l.exp());
    let vocab = ["BOS", "A", "B", "EOS"];
    println!(
        "Generated: {:?}",
        generate(&w, 0, 10)
            .iter()
            .map(|&i| vocab[i])
            .collect::<Vec<_>>()
    );
    for t in [0.5, 1.0, 2.0] {
        println!(
            "temperature={t}: P(next | A)={:.6?}",
            probabilities(&w[1], t)
        );
    }
    println!("A learned next-token table, not a trained Transformer.");
}
fn main() {
    println!("NN Chapter 12: Train a Next-Token Model and Decode");
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
