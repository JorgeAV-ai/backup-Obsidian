Contrastive learning trains representations by pulling similar (positive) pairs together and pushing dissimilar (negative) pairs apart in embedding space.

## What is it?

Contrastive learning is a self-supervised (or supervised) representation learning framework built on a simple geometric intuition: good representations should place semantically similar inputs close together and dissimilar inputs far apart. Rather than predicting labels directly, the model learns a distance structure over data points.

The setup requires defining **positive pairs** — two views of the same underlying example (e.g., two augmentations of the same image, a query and its relevant document) — and **negative pairs** — views from different examples. The model is trained so that the similarity between positive pairs is high relative to the similarity between negative pairs. The **temperature parameter** $\tau$ controls the sharpness of this contrast: smaller $\tau$ makes the model more discriminative but harder to train.

A key practical challenge is **hard negatives** — negatives that are superficially similar to the anchor but semantically different. Easy negatives (random samples) provide little gradient signal. Mining or constructing hard negatives is often the difference between a mediocre and a strong contrastive model. Milestones like **SimCLR** (Chen et al., 2020) showed that large batch sizes can serve as an implicit source of hard negatives, while **MoCo** (He et al., 2020) introduced a momentum-updated queue to decouple batch size from the number of negatives.

## How it works

![[basics_contrastive.png]]

[🔗 Open interactive Contrastive Playground](../../interactive/contrastive.html)

**InfoNCE loss** (for a positive pair $(i, j)$ within a batch of $N$ pairs, yielding $2N$ total augmented examples):

$$\mathcal{L} = -\log \frac{\exp\!\bigl(\text{sim}(z_i, z_j)/\tau\bigr)}{\displaystyle\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]}\, \exp\!\bigl(\text{sim}(z_i, z_k)/\tau\bigr)}$$

where:
- $z_i, z_j$ are the projected representations of the two views of example $i$
- $\text{sim}(u, v) = \frac{u^\top v}{|u|\,|v|}$ (cosine similarity)
- $\tau$ is the temperature scalar
- The denominator sums over all other examples in the batch (both positives and negatives of other pairs)

**Contrastive training loop (pseudocode):**

```
for batch in dataloader:
    # 1. Create two augmented views of each example
    x_a, x_b = augment(batch), augment(batch)

    # 2. Encode both views
    h_a, h_b = encoder(x_a), encoder(x_b)

    # 3. Project to contrastive space
    z_a, z_b = projector(h_a), projector(h_b)

    # 4. Compute pairwise cosine similarity matrix
    #    sim_matrix[i, j] = sim(z_a[i], z_b[j]) / tau
    sim_matrix = cosine_similarity(z_a, z_b) / tau

    # 5. Positives are on the diagonal (i, i)
    #    InfoNCE loss: cross-entropy where the diagonal is the correct class
    labels = range(N)
    loss = cross_entropy(sim_matrix, labels)

    # 6. Update
    loss.backward()
    optimizer.step()
```

**Step-by-step:**

1. **Augmentation** — generate two correlated views of each input (cropping, color jitter for images; paraphrasing, dropout masking for text).
2. **Encoding** — pass both views through a shared encoder to get representations $h$.
3. **Projection** — map $h$ through a small MLP projection head to get $z$ (the contrastive space). Representations $h$ are used downstream; the projector is discarded.
4. **Similarity computation** — build the $2N \times 2N$ similarity matrix, scaled by $1/\tau$.
5. **Loss** — for each anchor, treat its positive as the correct class among all $2N - 1$ candidates. Minimize InfoNCE.
6. **Hard negatives** — optionally, mine or weight negatives that have high similarity but are not positives, to sharpen the representation.

## Why it matters

Contrastive learning enables high-quality representations without labeled data. By defining what "similar" means through augmentations or known pairs, it sidesteps the need for expensive annotation. The resulting embeddings transfer well to downstream tasks (classification, retrieval, clustering) and form the backbone of modern dense retrieval systems, vision-language models (CLIP), and self-supervised pre-training pipelines.

## Used in

[[Hypothetical Document Embeddings (HyDE)]]
