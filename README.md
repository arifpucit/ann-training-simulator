# Teaching ANN Training with the Simulator

**Tool:** <https://arifpucit.github.io/ann-training-simulator/>
**Network:** 2 inputs → 2 hidden (ReLU) → 1 output (linear), no biases, Loss = (ŷ − y)²
**Covers:** the four steps of training: forward pass, loss, backpropagation, weight update; then optimizers

Every number below was produced by the simulator's own engine and transcribed, not
predicted. The default gradients were also checked with PyTorch autograd (see the end
of this guide). If a demo does not match, the deployed build is out of date.

---

## How the tool behaves

1. **Each question shows its full working.** Click **Next →** to move to the next
   question and **← Prev** to go back. Backpropagation questions list every chain-rule
   factor, and Figure A draws the path from the loss back to the weight.
2. **The network figure never runs ahead of the class.** z, a, ŷ, the loss and every
   gradient stay as **?** until you reach the question that computes them.
3. **Keys:** <kbd>→</kbd> next, <kbd>←</kbd> previous.
4. **Rounding.** Values are computed at full precision and shown to 4 decimals. Updated
   weights are rounded to 4 decimals, as students would round by hand, so every
   iteration can be checked with a calculator.
5. **Any values work.** Change the inputs, target, learning rate or any weight and click
   **▶ Start**; all 14 questions are rebuilt with the new numbers. **↺ Reset** restores
   the defaults.
6. **Deep links** open the simulator at a given question, e.g.
   `?step=8`, `?preset=dead&step=10`, `?part=2&opt=adam&step=3`.

One iteration is 14 questions in four colour-coded phases:

| # | Phase | Question |
|---|---|---|
| 1–3 | Forward pass | z₃, a₃ · z₄, a₄ · ŷ |
| 4 | Loss | (ŷ − y)², too high or too low? |
| 5 | Backpropagation | ∂Loss/∂ŷ, the factor shared by all six gradients |
| 6–7 | Backpropagation | ∂Loss/∂W₃₅, ∂Loss/∂W₄₅ (2 factors each) |
| 8–11 | Backpropagation | ∂Loss/∂W₂₄, W₁₄, W₁₃, W₂₃ (4 factors each) |
| 12 | Backpropagation | Sign and magnitude: what to do with each weight, and why |
| 13 | Weight update | W_new = W − lr × ∂Loss/∂W for all six weights |
| 14 | Weight update | Forward pass again: did the loss go down? |

---

## Demo 1 — The default walkthrough

**Goal:** one full training iteration, one question at a time. Defaults: x₁ = 0.35,
x₂ = 0.9, y = 0.5, lr = 0.1, W₁₃ = 0.1, W₂₃ = 0.8, W₁₄ = 0.4, W₂₄ = 0.6, W₃₅ = 0.3, W₄₅ = 0.9.

| Quantity | Value |
|---|---|
| z₃, a₃ | 0.755, 0.755 |
| z₄, a₄ | 0.68, 0.68 |
| ŷ | 0.8385 |
| Loss | 0.1146 (exact: 0.11458225) |
| ∂Loss/∂ŷ | 0.677 |

| Weight | Factors | ∂Loss/∂W | W_new (lr = 0.1) |
|---|---|---|---|
| W₂₄ | 0.677 × 0.9 × 1 × 0.9 | 0.5484 | 0.5452 |
| W₃₅ | 0.677 × 0.755 | 0.5111 | 0.2489 |
| W₄₅ | 0.677 × 0.68 | 0.4604 | 0.854 |
| W₁₄ | 0.677 × 0.9 × 1 × 0.35 | 0.2133 | 0.3787 |
| W₂₃ | 0.677 × 0.3 × 1 × 0.9 | 0.1828 | 0.7817 |
| W₁₃ | 0.677 × 0.3 × 1 × 0.35 | 0.0711 | 0.0929 |

Question 14 then shows **ŷ 0.8385 → 0.7154** and **Loss 0.1146 → 0.0464** after one update.

**Ask before moving on to Question 8:** *"Which factor of
∂Loss/∂W₂₄ have we already computed?"* The answer is ①, ∂Loss/∂ŷ. That reuse is the
whole efficiency argument for backpropagation.

**At Question 9 (W₁₄):** ask what changes compared with W₂₄. Only factor ④ (x₁ instead
of x₂); ①, ② and ③ are shared because both weights feed H₄.

---

## Demo 2 — The sign of the gradient

**Preset:** *ŷ too low (y = 1.2)*. Same weights and inputs; only the target changes.

```
ŷ = 0.8385, Loss = 0.1307, ∂Loss/∂ŷ = −0.723
gradients:  W₂₄ −0.5856   W₃₅ −0.5459   W₄₅ −0.4916
            W₁₄ −0.2277   W₂₃ −0.1952   W₁₃ −0.0759
```

Every gradient is now **negative**, so every weight goes **up**, and ŷ moves from
0.8385 to 0.978, toward 1.2. Loss after one update: 0.0493.

**Ask before moving on to Question 5:** *"The forward pass is identical to Demo 1. Which
number in the backward pass changes sign, and why does that flip all six?"*
∂Loss/∂ŷ = 2(ŷ − y) is the first factor of every product.

---

## Demo 3 — A dead ReLU blocks the gradient

**Preset:** *Dead ReLU (W₂₃ = −0.8)*.

```
z₃ = 0.35×0.1 + 0.9×(−0.8) = −0.685  →  a₃ = 0   (H₃ dead)
ŷ = 0.612, Loss = 0.0125 (exact 0.012544), ∂Loss/∂ŷ = 0.224
gradients:  W₂₄ 0.1814   W₄₅ 0.1523   W₁₄ 0.0706
            W₁₃ 0        W₂₃ 0        W₃₅ 0
```

Three gradients are exactly 0:

- **W₁₃ and W₂₃** because factor ③, ReLU′(z₃), is 0 (Question 10 says "this 0 wipes
  out the whole product").
- **W₃₅** because its second factor is a₃ = 0 (Question 6). A dead neuron also blocks
  learning on the weight *leaving* it.

H₄ is unaffected and keeps learning: Loss 0.0125 → 0.0072 after one update. In the
figure, H₃ turns grey and is labelled *dead (z ≤ 0)*.

---

## Demo 4 — Learning rate too large

**Preset:** *lr too large (lr = 1.0)*. Gradients are identical to Demo 1; only the step size changes.

```
new weights: W₁₃ 0.0289  W₂₃ 0.6172  W₁₄ 0.1867  W₂₄ 0.0516  W₃₅ −0.2111  W₄₅ 0.4396
ŷ: 0.8385 → −0.0703     Loss: 0.1146 → 0.3252
```

The prediction jumps **past** the target 0.5 and lands farther away on the other side,
so the loss **rises**. Question 14 says so and names the cause. W₃₅ changes sign, so
from iteration 2 its edge is drawn in orange.

Compare with lr = 0.1 (Demo 1), where the loss fell to 0.0464.

---

## Demo 5 — Repeat the loop

From the defaults, finish Question 14 and click **Start iteration 2 →**, or click
**⏩ Train 10 epochs/iterations**. The loss at the start of each iteration:

```
0.1146 → 0.0464 → 0.021 → 0.01 → 0.005 → 0.0025 → 0.0013 → 0.0007 → 0.0004 → 0.0002 → 0.0001
```

Figure 2 plots this curve. Point out that each point is one pass through all four steps.

---

## Demo 6 — Your own numbers

Ask the class for values, type them in, and click **▶ Start**. The questions, the figures
and the explanations (which gradient is strongest and why, whether a neuron is dead,
whether the loss fell or rose) are all rebuilt from the new numbers. If both hidden
neurons are dead, or the prediction is already exact, Question 12 explains why all six
gradients are 0.

---

## Side note — Batch, Mini-Batch and Stochastic GD

Shown with Question 13 and in the side-note figure. The simulator trains on **one**
sample, so the three variants coincide. With N samples, backprop runs per sample exactly
as shown, the gradients are averaged over the batch, and then one update is applied:

| Variant | Samples per update | Updates per epoch |
|---|---|---|
| Batch GD | all N | 1 |
| Mini-Batch GD | e.g. 32 | N / 32 |
| Stochastic GD | 1 | N |

---

## Part 2 — Optimizers (after Part 1)

Open the **Part 2** tab only after one full iteration of Part 1. The banner makes the
point: an optimizer changes **only step 4**. The defaults are W = 0.5, gradient = −0.4,
lr = 0.1, ε = 0.01, with the carried-over state shown below. Pick each optimizer from
the dropdown:

| Optimizer | State before → after | W_new | Direction |
|---|---|---|---|
| Plain GD | none | 0.54 | up |
| SGD + Momentum | v 0.2 → 0.14 | 0.486 | **down** |
| AdaGrad | acc 0.5 → 0.66 | 0.5489 | up |
| RMSprop | v 0.3 → 0.2986 | 0.572 | up |
| Adam | m 0.1 → 0.05, v 0.2 → 0.19996 | 0.4891 | **down** |

Figure 4 adds each optimizer as you finish it.

**Ask before moving on to Momentum's W_new question:** *"The gradient is negative. Which way will W
move?"* Most students say up. It moves down, because the old velocity (+0.2) still
outweighs the new gradient contribution (−0.04).

Figure 5 runs all five optimizers for 30 updates on the default Part 1 network, starting
from zero optimizer state. It is one short run on one sample, not a ranking of optimizers.

---

## Verifying with PyTorch

Run this once in class so students trust the tool. Autograd should produce the default
gradients:

```python
import torch

x = torch.tensor([0.35, 0.90])
y = torch.tensor(0.5)
W_ih = torch.tensor([[0.1, 0.8],     # row 0 = H3: [W13, W23]
                     [0.4, 0.6]],    # row 1 = H4: [W14, W24]
                    requires_grad=True)
W_ho = torch.tensor([0.3, 0.9], requires_grad=True)   # [W35, W45]

a = torch.relu(W_ih @ x)       # forward pass: hidden layer
y_hat = W_ho @ a               # output, no activation
loss = (y_hat - y) ** 2        # loss
loss.backward()                # backpropagation

print(f"y_hat = {y_hat.item():.4f}   loss = {loss.item():.4f}")
print("dL/dW13, dL/dW23 =", [round(v, 4) for v in W_ih.grad[0].tolist()])
print("dL/dW14, dL/dW24 =", [round(v, 4) for v in W_ih.grad[1].tolist()])
print("dL/dW35, dL/dW45 =", [round(v, 4) for v in W_ho.grad.tolist()])
```

Output (PyTorch 2.11):

```
y_hat = 0.8385   loss = 0.1146
dL/dW13, dL/dW23 = [0.0711, 0.1828]
dL/dW14, dL/dW24 = [0.2133, 0.5484]
dL/dW35, dL/dW45 = [0.5111, 0.4604]
```

`loss.backward()` does in one line what Questions 5–11 do by hand.

---

## Publishing

The tool is a single `index.html` with no dependencies. In the repository settings,
enable **Pages → Deploy from branch → main / (root)**.

Instructors: Muhammad Rauf Butt (FCCU) and Muhammad Arif Butt (PU)
