### 📌 Pinned Entry

<h4 style="margin-bottom:0;">First Weeks into Learning Rust</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-22 · Rust Learning week 3</em>
</p>
So I started on this journey realizing I had no idea what the functions I was calling were doing. I'd apply a softmax to some values that seemed very random to me following documentation, writing in my notes what intuitively everything did but I truely didn't understand what I was writing.

I've been working on MSE, matrix multiplication, dot products, relu, and then working that together to create a super simply feed forward network.

Basically I'm building myself up from the basics so that I have a strong understanding of how everything works.

It worked super nicely because I'd build in the bias function, the ReLu, and dot products and then on the very next mini challenge, use my old code for the new problem.

```Rust
pub fn matmul(...) { ...}
pub fn add_bias(...) {...}
pub fn relu(...) {...}

pub fn dense_forward(
    x: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
) -> Option<Vec<Vec<f32>>> {
    if x.is_empty() || w.is_empty() || b.is_empty() {
        return None;
    }

    let z = Self::matmul(x, w);
    let z_b = Self::add_bias(z?, b);
    let y = Self::relu(z_b?);
    Some(y)
```

How I've been learning thusfar:
I've been leaning on AI to roadmap my progression from skillset to skillset and have it generate LeatCode style files for me to build ML stuff. It's been working great but I'm hoping I can reach out to someone more senior for some guidance on how they'd go from here.

```Rust
fn main() {
  println!("\x1b[1m\x1b[35mNN Runner: Dense Layer Forward\x1b[0m");

  // Identity-like layer, no bias
  print_result(
      "Test Case 1",
      vec![vec![1.0, 2.0]],
      vec![vec![1.0, 0.0], vec![0.0, 1.0]],
      vec![0.0, 0.0],
      Some(vec![vec![1.0, 2.0]]),
  );

  // ...
}
```

<h4 style="margin-bottom:0;">First Hours into Learning Rust</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-11 · Rust Learning Day 1</em>
</p>

> Lot's of confusion but my focus has been to avoid LLM code like it's the plague. Autocorrect is off and so is Copilot.
>
> I'm reading through the amazing Rust documentation along with doing Leatcode problems.
>
> As for the use of AI, I've been using Perplexity.ai for choosing Leatcode problems and writing tests for them exlcusively.
>
> All code is human written. I've loved the look of Rust code and I always imitate it in Python anyways so I'm hoping this keeps up. It's a little bit of a learning curve.
> As long as I keep challenging myself and going through the docs I'll eventually learn.
>
> The syntax too is starting to make sense but I have to think about it.
>
> ```Rust
> Vec<i32> // Right cos Vector = Array = List and i32 interger 32 bits
> for i in 0..nums.len() // 0..nums like the ... in math and .len() is an improvement
> -> Vec<i32> // is also phenominal because no typing library is required
> return // Early return and auto returning is amazing
> ```
>
> Still wrapping my head around som errors with pointers and borrowing. Anyways did two Leatcode problems in rust.
>
> ```Rust
> impl Solution {
>     pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
>         for i in 0..nums.len() {
>             for j in (i+1)..nums.len() {
>                 if nums[i] + nums[j] == target {
>                     return vec![i as i32, j as i32];
>                 }
>             }
>         }
>     }
> }
>
> // Which I optimized using hashmaps to
> use std::collections::HashMap;
> let mut map: HashMap<i32, usize> = HashMap::new();
>
> for (i, &val) in nums.iter().enumerate() {
>     let mut complement = target - val;
>     if let Some(&j) = map.get(&complement) {
>         return vec![j as i32, i as i32];
>     }
>     else {
>         map.insert(val, i);
>     }
> }
>
>
> // &
>
> impl Solution {
>     pub fn is_palindrome(s: String) -> bool {
>         let filtered: String = s
>             .to_lowercase()
>             .chars()
>             .filter(|c| c.is_alphanumeric())
>             .collect();
>
>         let reversed: String = filtered
>             .chars()
>             .rev()
>             .collect();
>
>         filtered == reversed
>     }
> }
> ```

<h4 style="margin-bottom:0;">Steping it up—Leatcode Medium Difficulty</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-11 · Rust Learning Day 3</em>
</p>

Today I worked on a medium difficulty leatcode problem. It went pretty well but I learned some new syntax. This was a command I used quite a lot in Python but getting familiar with it in Rust was nice.

```Rust
let x = 3
let y = 4
x.max(y)
// Output: 4
```

I had already solved the Leatcode problem beforehand but the solution used max and it greatly simplified my code.

I'm also noticing that writing in Rust has already started becoming very natural to me. It's a lot like python in certain ways at least the way I write my Python code—with a little too much perfectionism.

<h4 style="margin-bottom:0;">Notes pre-writing</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-12 · Rust Learning Day 3</em>
</p>

```Rust
// ❌let mut log: Vec<i32> = vec![]; because idx will be usize?!?!
// so instead:
let mut log: Vec<usize> = vec![];
// ...
for (idx, x)  in nums.iter().skip(1).enumerate() {
  // ...
  log.push(idx + 1); // now works
}
```

Then I'm starting to learn about the borrowing system which feels confusing right now but I'm already starting to see the logic behind it.

```Rust
let mut current: i32 = nums[0];
for (idx, x)  in nums.iter().skip(1).enumerate() {
  if current < *x {
    // ... prevents errors by adding *x because current is i32 and
    // x is a reference to a value
  }
}
```

But there's a better solution

```Rust
let mut current: i32 = nums[0];
for (idx, &x) in nums.iter().skip(1).enumerate() {
  if current < x {
    // ... Here by borrowing we avoid the missmatched type issue
    // because we borrow from x
  }
}
```

<h4 style="margin-bottom:0;">Moving into Softmax</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-23 · Rust Learning week 4</em>
</p>

Softmax has kinda just been this mysterious function I'd call to turn this seemingly abitrary numbers into probablilities. Now I understand how it funcitons. It's really clean and also cool that it uses `e` a lot because that means the derative will be interesting to learn about later on down the road for maybe backprop.

So for softmax, the problem I did was a row by row softmax calculation. I wrote a general function, and then this sub function:

```Rust
pub fn softmax_row(xs: &Vec<f32>) -> Vec<f32> {
    let mut denominator = 0.0_f32;
    let mut out = Vec::new();

    for j in xs {
        denominator += j.exp();
    }

    for i in xs {
        out.push( i.exp() / denominator);
    }

    out
}
```

I decided to calculate the denominator first because I realized that exponential equations can take up a lot of memory and down the road that will be a concern once the numbers are larger. Basically I pulled up Victor Zhou's amazing article and implemented in Rust. Super cool dude! Basically copied this equation.

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

Other learnings:

```Rust
Vec::new(); // I'm prefering this over vec![]; currently
for row in &matrix { ... } // because row inherits the val from matrix
// it must be borrowed instead of just for row in matrix
let x = 43.32_f32; // Learned that I can just do _f32 and it's pretty

```

<h4 style="margin-bottom:0;">`with_capacity`</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-26 · Rust Learning week 4</em>
</p>

I haven't updated this in a few days now. I've been networking and devleoping silently for a few days.

I thought I'd write about `Vec::with_capacity();`. It makes code so much more readable.

```Rust
let cols: usize = b.len();
let mut new_b = Vec::with_capacity(cols);

for j in 0..cols {
  new_b.push( b[j] - lr * db[j] );
}

// Instead of
let cols = b.len();
let mut new_b = vec![0.0; cols];

for j in 0..cols {
    new_b[j] = b[j] - lr * db[j];
}

```

Personally I find the first version signficantly more readable.

Also I ran into a few issues with the DenseBackward problem but I was able to fix most of them. Simple things like cloning variables.

```Rust
let dW = Self::matmul(Self::transpose(x), dL_dZ.clone());
let dX = Self::matmul(dL_dZ.clone(), Self::transpose(w));
```

<h4 style="margin-bottom:0;">Some updates on how I'm learning</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-27 · Rust Learning week 4</em>
</p>

I've really been leveraging AI to learn Rust very quickly. I want to do something like for example create an instance using the attributes from a class and call functions inside a different class and using this example I can easily translate the code into what it'd look like in mine.

```Rust
struct Car {
    color: String,
    brand: String,
    year: u32,
}

impl Car {
    fn honk(&self, emotion: &str) {
        println!("The {} {} car honks {}!", self.color, self.brand, emotion);
    }
}

fn main() {
    let car = Car {
        color: String::from("red"),
        brand: String::from("Toyota"),
        year: 2020,
    };

    println!("Car color: {}", car.color);
    println!("Car year: {}", car.year);

    car.honk("angrily");
    car.honk("happily");
}
```

<h4 style="margin-bottom:0;">Built a multi-neuron, 2d net</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-27 · Week 5</em>
</p>

Still yet to add randomness to the wieghts and bias tuning process. Today I discovered this was really really hurting performance. Basically every neuron had the exact same wieght and bias. Basically a symetry, The ideal would be for each to symbolize a different trend and pattern in the data and then together they'll fidn the pattern.

I'm planning on doing a new activity where I do simply a bunch of exercises of just initializing a bunch of different random wieghts and biases. After all the whole goal of this project is building my ML understanding, not just speed with random functions in libraries.

I've been using this really cool browser extension lately for doing math. The `\partial` command has come in handy every day constantly. I've found once I understood the math and solved derivitives myself, things started making a lot more sense.

I kept looking at these solutions using "delta" all the while thinking how it's strange how I'm doing diff xi and diff for d_w and d_b instead of just setting diff as a universal term or something. It just so happened that I glanced at a solution with calculations for delta and realized it's just the first and second derivitive.

$$
\delta_j = \hat{y}_j - y_j
$$

I then calculated these manually for the second layer for learning purposes.

$$
\frac{\partial L}{\partial W^{(2)}_{h j}} = \delta_j \, a^{(1)}_h
\qquad
\frac{\partial L}{\partial b^{(2)}_j} = \delta_j
$$

$$
\delta^{(1)}_h
= \left( \sum_j W^{(2)}_{h j} \, \delta_j \right) \text{ReLU}'(z^{(1)}_h)
$$


<h4 style="margin-bottom:0;">Returning to Rust</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-11-8 · Week 1.6</em>
</p>
I've got so sick of this AI building treadmill. You build projects, learn nothing for what? My EEG project stalled, I tried a bunch of solutions, my mentor told me to just churn out as many different attempts as I could with AI, but I don't want to do that. I've officially downloaded RustRover, and I'm going deeper into math. I've rejoined the school math team and I intend to stay there. I'm studying for the CCC, last year one point off from the top 95th percentile, and I don't intend to let that happen this year. I've just had to readjust to Rust again. My first lines comically being:

```Rust
fn main(){
    print!("hello");
    println!(" world");
    print!("world");
}
```

So yeah then later today, just warming up again, I've just started with a simple perceptron code:
{\fontsize{12}{14.4}\selectfont $\begin{cases}w_0x_0+w_1x_1+b\ge0 & 1\\ w_0x_0+w_1x_1+b\:<0 & 0\end{cases}$
```Rust
fn predict(&self, x: [f64; 2]) -> i32 {
    (self.w[0] * x[0] + self.w[1] * x[1] + self.b >= 0.0) as i32
}
```

