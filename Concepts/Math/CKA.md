Centered Kernel Alignment (CKA) is a similarity metric that compares the representational geometry of two sets of neural network activations, revealing whether different layers (or models) encode similar information.

## What is it?

When analyzing deep networks, a natural question is: "Do layer 5 and layer 10 learn the same thing?" Answering this requires comparing entire representation matrices, not individual vectors. Naive approaches like cosine similarity between individual neuron activations fail because representations are distributed — the same information can be spread across neurons in completely different ways. Two layers might encode identical structure but with permuted or rotated neuron axes, and cosine similarity would miss this entirely.

**CKA** solves this by comparing the *kernel matrices* (pairwise similarity structures) induced by each layer's activations over a set of inputs. If two layers assign the same relative similarities to all input pairs, CKA will be high, regardless of how the information is arranged across neurons. Formally, it is built on the **Hilbert-Schmidt Independence Criterion (HSIC)**, a kernel-based statistical test of dependence between two random variables. HSIC measures how much the pairwise structure in one representation agrees with the pairwise structure in another.

A crucial property of CKA is its **invariance to orthogonal transformations** and **isotropic scaling**. This means that if one layer's representation is simply a rotation or uniform rescaling of another's, CKA correctly identifies them as equivalent. This invariance is exactly what makes it more meaningful than per-neuron comparisons for understanding deep network internals.

## How it works

![[basics_cka.png]]

**CKA formula:**

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

where:
- $K = X X^\top$ and $L = Y Y^\top$ are the kernel (Gram) matrices for activation matrices $X$ (layer 1) and $Y$ (layer 2), computed over $n$ input examples
- HSIC is the Hilbert-Schmidt Independence Criterion

**HSIC (with linear kernels and centering):**

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \operatorname{tr}(\tilde{K}\, \tilde{L})$$

where $\tilde{K} = H K H$ is the centered kernel matrix, and $H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centering matrix.

**Step-by-step computation:**

1. **Collect activations** — run $n$ input examples through the network. Record the activation matrix $X \in \mathbb{R}^{n \times p}$ for layer 1 and $Y \in \mathbb{R}^{n \times q}$ for layer 2.
2. **Compute kernel matrices** — $K = X X^\top$, $L = Y Y^\top$ (both $n \times n$).
3. **Center** — $\tilde{K} = H K H$, $\tilde{L} = H L H$.
4. **Compute HSIC** — $\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \operatorname{tr}(\tilde{K}\,\tilde{L})$.
5. **Normalize** — divide by the geometric mean of self-HSIC values to get CKA in $[0, 1]$.

**Pseudocode:**

```
def linear_CKA(X, Y):
    """X: (n, p) activations from layer 1
       Y: (n, q) activations from layer 2"""
    # Center activations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # HSIC with linear kernel (simplified)
    hsic_xy = ||Y^T X||_F^2          # Frobenius norm squared
    hsic_xx = ||X^T X||_F^2
    hsic_yy = ||Y^T Y||_F^2

    return hsic_xy / sqrt(hsic_xx * hsic_yy)
```

**Interpreting the result:**
- $\text{CKA} \approx 1$ — the two layers capture nearly identical representational structure (potential redundancy).
- $\text{CKA} \approx 0$ — the layers encode unrelated information.

## Why it matters

CKA provides a principled, invariant way to compare what different layers or models have learned. In transformer analysis, plotting the CKA matrix across all layer pairs reveals which layers are redundant ($\text{CKA} \approx 1$) and which contribute genuinely new representations. This insight directly motivates architectural efficiency improvements: if consecutive layers are nearly identical, some can be skipped or pruned without meaningful loss, reducing computation while preserving performance.

## Used in

[[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]
