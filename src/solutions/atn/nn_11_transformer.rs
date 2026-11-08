/*
NN Chapter 11. Assemble a Complete Tiny Decoder Block
[src/bin/nn_11_transformer.rs]

Goal:
Build a deterministic two-head causal Transformer forward pass with positional
encoding, output projection, residuals, LayerNorm, a feed-forward network and logits.

Why this matters for neural networks:
Attention is one component, not the whole model. This chapter assembles the pieces
and tests their contracts. Concatenating heads changes width; the output projection
returns it to d_model so the residual addition is dimensionally valid.

Setup and dimensions:
Vocabulary V=4, width D=4, heads H=2, dk=dv=2, FFN width F=8.
Input token IDs: T; embeddings X: T x 4; each head: T x 2.
Concatenated heads: T x 4; output logits: T x 4.
Weights are fixed deterministic sine values, not random or pretrained.

Equations and gradient rules:
X[t]=sqrt(D)*E[token[t]]+PE[t]
PE(pos,2i)=sin(pos/10000^(2i/D)); PE(pos,2i+1)=cos(pos/10000^(2i/D))
head_h=Attention(X Wq_h, X Wk_h, X Wv_h)
M=Concat(head_0,head_1) Wo
U=LayerNorm(X+M)
F=ReLU(U W1+b1) W2+b2
Z=LayerNorm(U+F); logits=Z W_vocab
LayerNorm(row)=(row-mean)/sqrt(variance+epsilon)
Here epsilon=1e-5, gamma=1, beta=0; b1=0.05 and b2=0.
This is POST-NORM, not the pre-norm arrangement of many later models.

Implementation steps:
1. Add positions to token embeddings before computing Q/K/V.
2. Run two separately projected heads, each with a causal mask.
3. Concatenate features, project back to D, then add the residual.
4. Normalize each token across its feature dimension, not across tokens.
5. Apply the same FFN at every position, another residual and normalization.
6. Project to vocabulary logits; convert to probabilities only when needed.

Important limitations:
Forward-only and untrained. No dropout, learned LayerNorm affine parameters,
padding mask, cross-attention, KV cache or full Transformer training loop.
Changing a FUTURE token must not change any earlier logits; changing a past
token may affect later logits. Tests check both behaviors.

The jump to the next idea:
Architecture supplies logits. Chapter 12 makes the next-token objective,
parameter learning and autoregressive decoding explicit with a small trainable model.

How to practice:
- Read the brief and the function hints before the completed implementation.
- Reimplement one function at a time, preserving its signature.
- Run the named checks after each change. Compare numerical values, not just colors.
- These files are complete solutions, not unfinished TODO stubs.

Run:
  cargo run --bin nn_11_transformer
  cargo test --bin nn_11_transformer
Or, independently:
  rustc --edition=2021 nn_11_transformer.rs -o lesson
  ./lesson

Notes:
- Standard library only, Rust 2021. f64 is deliberate for gradient checks.
- Finite-difference checks use epsilon=1e-6 and absolute tolerances.
- The runner catches each assertion to report every case, then exits nonzero
  if any case failed. A limitation demonstration is labeled, not scored as a bug.
- NO_COLOR=1 disables terminal colors. Redirected output is plain text.

Further reading (notation and theory):
https://arxiv.org/html/1706.03762
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

type Mat = Vec<Vec<f64>>;
/// Hints: A is m x k and B is k x n. Contract k; preserve m and n. Check rectangularity before indexing.
fn matmul(a: &Mat, b: &Mat) -> Mat {
    assert!(!a.is_empty() && !b.is_empty() && !a[0].is_empty() && !b[0].is_empty());
    assert!(a.iter().all(|r| r.len() == a[0].len()));
    assert!(b.iter().all(|r| r.len() == b[0].len()));
    assert_eq!(a[0].len(), b.len());
    (0..a.len())
        .map(|i| {
            (0..b[0].len())
                .map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}
/// Hints: Output row j is input column j. Transposition changes layout, not the underlying values.
fn transpose(a: &Mat) -> Mat {
    (0..a[0].len())
        .map(|j| (0..a.len()).map(|i| a[i][j]).collect())
        .collect()
}
/// Hints: Subtract the largest allowed score, exponentiate, then normalize. A masked -infinity becomes probability zero.
fn softmax(row: &[f64]) -> Vec<f64> {
    assert!(!row.is_empty() && row.iter().any(|x| x.is_finite()));
    assert!(row.iter().all(|x| x.is_finite() || *x == f64::NEG_INFINITY));
    let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|x| x / sum).collect()
}
/// Hints: Compute QK^T, divide by sqrt(dk), mask BEFORE softmax, then multiply normalized rows by V.
fn attention(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> (Mat, Mat) {
    assert_eq!(k.len(), v.len());
    assert!(!q.is_empty() && !k.is_empty());
    assert!(k.iter().all(|r| r.len() == q[0].len()));
    let scores = matmul(q, &transpose(k));
    let scale = (q[0].len() as f64).sqrt();
    let a: Mat = scores
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let s: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(j, x)| {
                    if causal && j > i {
                        f64::NEG_INFINITY
                    } else {
                        x / scale
                    }
                })
                .collect();
            softmax(&s)
        })
        .collect();
    (matmul(&a, v), a)
}

/// Hints: These deterministic sine values are fixtures, not learned parameters. Offsets distinguish separate projection matrices.
fn weights(rows: usize, cols: usize, offset: usize) -> Mat {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| 0.3 * (((i * cols + j + offset + 1) as f64) * 0.7).sin())
                .collect()
        })
        .collect()
}
/// Hints: Residual operands must have the same shape; a zip without shape checks could silently truncate.
fn add(a: &Mat, b: &Mat) -> Mat {
    {
        assert_eq!(a.len(), b.len());
        for (ar, br) in a.iter().zip(b) {
            assert_eq!(ar.len(), br.len())
        }
    }
    a.iter()
        .zip(b)
        .map(|(ar, br)| ar.iter().zip(br).map(|(x, y)| x + y).collect())
        .collect()
}
/// Hints: Compute mean and population variance within EACH token row. Epsilon stays inside the square root.
fn layernorm(x: &Mat) -> Mat {
    assert!(!x.is_empty() && x.iter().all(|r| !r.is_empty()));
    x.iter()
        .map(|row| {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64;
            row.iter()
                .map(|v| (v - mean) / (var + 1e-5).sqrt())
                .collect()
        })
        .collect() // gamma=1, beta=0
}
/// Hints: Follow the shape chain in Setup. Preserve residual width when concatenating and projecting heads.
fn model(tokens: &[usize]) -> (Mat, Vec<Mat>) {
    assert!(!tokens.is_empty() && tokens.iter().all(|&t| t < 4));
    let d = 4;
    let embeddings = weights(4, d, 1);
    let x: Mat = tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            (0..d)
                .map(|j| {
                    let angle = pos as f64 / 10000_f64.powf((2 * (j / 2)) as f64 / d as f64);
                    let pe = if j % 2 == 0 { angle.sin() } else { angle.cos() };
                    (d as f64).sqrt() * embeddings[tok][j] + pe
                })
                .collect()
        })
        .collect();
    let mut heads = Vec::new();
    let mut maps = Vec::new();
    for head in 0..2 {
        let q = matmul(&x, &weights(d, 2, 10 + head * 30));
        let k = matmul(&x, &weights(d, 2, 20 + head * 30));
        let v = matmul(&x, &weights(d, 2, 30 + head * 30));
        let (out, a) = attention(&q, &k, &v, true);
        heads.push(out);
        maps.push(a);
    }
    let joined: Mat = (0..x.len())
        .map(|i| {
            let mut row = heads[0][i].clone();
            row.extend_from_slice(&heads[1][i]);
            row
        })
        .collect();
    let projected = matmul(&joined, &weights(d, d, 80));
    let u = layernorm(&add(&x, &projected));
    let mut ff = matmul(&u, &weights(d, 8, 100));
    for row in &mut ff {
        for v in row {
            *v = (*v + 0.05).max(0.0);
        }
    }
    let ff = matmul(&ff, &weights(8, d, 140));
    let z = layernorm(&add(&u, &ff));
    let logits = matmul(&z, &weights(d, 4, 180));
    (logits, maps)
}

fn cases() -> Vec<Case> {
    vec![
        ("Logits and both attention maps have correct shapes", || {
            let (z, m) = model(&[0, 1, 2]);
            assert_eq!(z.len(), 3);
            assert!(z.iter().all(|r| r.len() == 4));
            assert_eq!(m.len(), 2);
            assert!(m
                .iter()
                .all(|a| a.len() == 3 && a.iter().all(|r| r.len() == 3)));
        }),
        ("Every head is causal and row-normalized", || {
            let (_, m) = model(&[0, 1, 2]);
            for a in m {
                for i in 0..3 {
                    close(a[i].iter().sum(), 1.0, 1e-12);
                    for j in i + 1..3 {
                        close(a[i][j], 0.0, 0.0)
                    }
                }
            }
        }),
        ("Future token cannot change either earlier position", || {
            let (a, _) = model(&[0, 1, 2]);
            let (b, _) = model(&[0, 1, 3]);
            for t in 0..2 {
                for j in 0..4 {
                    close(a[t][j], b[t][j], 1e-12)
                }
            }
        }),
        ("Prefix and longer-sequence logits agree", || {
            let (a, _) = model(&[0, 1]);
            let (b, _) = model(&[0, 1, 2, 3]);
            for t in 0..2 {
                for j in 0..4 {
                    close(a[t][j], b[t][j], 1e-12)
                }
            }
        }),
        (
            "LayerNorm yields zero mean and epsilon-adjusted variance",
            || {
                let z = layernorm(&vec![vec![1.0, 2.0, 3.0, 4.0]]);
                close(z[0].iter().sum::<f64>() / 4.0, 0.0, 1e-12);
                close(
                    z[0].iter().map(|v| v * v).sum::<f64>() / 4.0,
                    1.25 / (1.25 + 1e-5),
                    1e-12,
                );
            },
        ),
        ("Constant-row LayerNorm is finite zero", || {
            assert_eq!(layernorm(&vec![vec![3.0; 4]]), vec![vec![0.0; 4]])
        }),
        ("Past-token change affects a later output", || {
            let (a, _) = model(&[0, 1, 2]);
            let (b, _) = model(&[3, 1, 2]);
            assert!(a[2].iter().zip(&b[2]).any(|(x, y)| (x - y).abs() > 1e-8));
        }),
        ("Vocabulary probabilities normalize", || {
            let (z, _) = model(&[0, 1, 2]);
            for row in z {
                close(softmax(&row).iter().sum(), 1.0, 1e-12)
            }
        }),
    ]
}
fn demo() {
    let (z, maps) = model(&[0, 1, 2]);
    let targets = [1, 2, 3];
    for (h, a) in maps.iter().enumerate() {
        println!("head {h} attention={a:.5?}");
    }
    let loss = -(0..3).map(|t| softmax(&z[t])[targets[t]].ln()).sum::<f64>() / 3.0;
    println!("Logits={z:.5?}\nMean next-token NLL={loss:.6}");
    println!("All weights are deterministic and UNTRAINED. These logits do not demonstrate language competence.");
}
fn main() {
    println!("NN Chapter 11: Assemble a Complete Tiny Decoder Block");
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
