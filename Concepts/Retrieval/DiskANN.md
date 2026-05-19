DiskANN is a graph-based approximate nearest neighbor index that stores the graph on SSD instead of RAM, enabling billion-scale search on a single commodity machine.

## What is it?

DiskANN (NeurIPS 2019, Microsoft Research) solves the problem of approximate nearest neighbor search (ANNS) at billion-point scale. Prior graph-based methods like HNSW and NSG require the entire index to fit in RAM — at 1 billion 128-dim float32 vectors, that's ~512 GB. DiskANN breaks this constraint by keeping the graph on an SSD while keeping only compressed representations (PQ codes) in RAM to guide search.

The core data structure is the **Vamana graph**: a directed graph over the dataset where each node points to at most $R$ neighbors chosen to balance proximity and long-range connectivity. Vamana is more aggressive than HNSW at pruning redundant edges, which reduces the graph's degree and makes SSD I/O manageable.

**pgvectorscale** (Timescale) uses DiskANN as its underlying index, making billion-scale vector search available inside PostgreSQL. See [pgvectorscale & StreamingDiskANN](#pgvectorscale--streamingdiskann) for implementation details.

## How it works

### Vamana: graph construction

The graph is built by inserting each point one at a time. For each new point, `GreedySearch` finds its approximate neighborhood, then `RobustPrune` selects which edges to keep.

**GreedySearch** — finds the $k$ approximate nearest neighbors of a query starting from entry point $s$:

```python
def GreedySearch(G, s, query, k, L):
    candidates = {s}      # active frontier
    visited = set()       # already expanded nodes

    while (candidates - visited) != empty:
        p_star = argmin_{p in candidates - visited} dist(p, query)
        visited.add(p_star)
        candidates.update(G.neighbors(p_star))

        if len(candidates) > L:
            candidates = top_L_closest(candidates, query)  # prune frontier

    return top_k(candidates, query), visited
```

> [!info]- Why L > k?
> `L` is the beam width — larger than $k$ to avoid local minima. The search expands more nodes than needed and returns the best $k$ at the end. Higher $L$ → higher recall, higher latency.

**RobustPrune** — selects at most $R$ edges from a candidate set, pruning neighbors that have a "shortcut" already covered by a closer node:

```python
def RobustPrune(G, p, candidates, alpha, R):
    candidates = candidates | set(G.neighbors(p))
    G.remove_all_edges_from(p)

    while candidates and len(G.neighbors(p)) < R:
        p_star = argmin_{c in candidates} dist(p, c)   # closest remaining
        G.add_edge(p, p_star)

        # prune any candidate p' that p_star already "covers"
        for p_prime in list(candidates):
            if alpha * dist(p_star, p_prime) <= dist(p, p_prime):
                candidates.remove(p_prime)
```

> [!info]- The alpha parameter
> $\alpha \geq 1$ controls pruning aggressiveness. If $\alpha \cdot d(p^*, p') \leq d(p, p')$, it means $p^*$ is already a good stepping stone to $p'$, so keeping a direct edge from $p$ to $p'$ is redundant. $\alpha = 1$ gives the tightest pruning; $\alpha > 1$ is more permissive and retains longer-range edges — useful for improving recall at the cost of higher degree.

**Full Vamana construction:**

```python
def Vamana(P, R, L, alpha):
    G = random_R_regular_graph(P)   # random init for connectivity
    s = medoid(P)                    # entry point = geometric center

    for p in random_permutation(P):
        _, visited = GreedySearch(G, s, p, 1, L)
        RobustPrune(G, p, visited, alpha, R)

        for q in G.neighbors(p):    # maintain approximate symmetry
            if deg(q) + 1 > R:
                RobustPrune(G, q, G.neighbors(q) | {p}, alpha, R)
            else:
                G.add_edge(q, p)
    return G
```

---

### SSD layout: one sector per node

The key engineering insight is storing each node as a **single aligned sector** on the SSD:

```
[ full vector (128 × 4 bytes) | neighbor IDs | neighbor PQ codes ]
```

During search, each graph hop = **one sector read** from SSD. The neighbor PQ codes are stored inline so the next hop's candidates can be scored in RAM (with approximate distances) before issuing the next round of SSD reads — this is the **beam search over SSD**:

```python
def SSD_Search(query, G_ssd, pq_cache, k, L, beam_width):
    # PQ cache is the only thing in RAM
    candidates = [(dist_pq(query, s), s)]  # start at medoid

    while not converged:
        # 1. Pick best beam_width unexpanded nodes
        to_expand = top_beam(candidates, expanded=False)

        # 2. Batch-read their sectors from SSD (amortize I/O)
        sectors = SSD.read_batch([n.id for n in to_expand])

        # 3. Compute exact distances, update candidate list
        for sector in sectors:
            exact_dist = dist(query, sector.full_vector)
            for nbr_id, nbr_pq in zip(sector.neighbor_ids, sector.neighbor_pq):
                approx_d = dist_pq(query, nbr_pq)
                candidates.push((approx_d, nbr_id))  # PQ score for ordering

        # 4. Re-rank candidates by exact dist, keep top L
        candidates = top_L_by_exact(candidates, L)

    return top_k(candidates, k)
```

> [!info]- Product Quantization (PQ) for in-memory scoring
> PQ compresses a $d$-dimensional vector into a short code (e.g., 8–16 bytes) by splitting it into $m$ subvectors and quantizing each independently with a small codebook. Distance to the query can be approximated without decompressing, using precomputed lookup tables. At 128-dim with $m=16$: 16 bytes/vector vs 512 bytes — **32× compression** — so billions of PQ codes fit in 64 GB RAM.

---

### Why graph-based beats IVF at high recall

IVF (Inverted File Index) partitions the space into $k$ clusters. To reach high recall, you must probe many clusters — at 95%+ recall, IVF latency spikes. Graph-based search navigates directly toward the query following edges, requiring far fewer distance computations in the high-recall regime.

| Method | 95% recall latency | Memory (1B vecs) |
|---|---|---|
| IVF-PQ (FAISS) | ~50 ms | ~64 GB |
| HNSW | <1 ms | ~512 GB (full RAM) |
| **DiskANN** | **<3 ms** | **64 GB RAM + SSD** |

## Benchmarks

Standard ANNS benchmark suite used to evaluate DiskANN (and most competing methods):

| Dataset | Vectors | Dim | Type | Distance | Domain |
|---|---|---|---|---|---|
| **SIFT1M** | 1M | 128 | uint8 | L2 | SIFT image features |
| **GIST1M** | 1M | 960 | float32 | L2 | GIST image descriptors |
| **DEEP10M** | 10M | 96 | float32 | L2 | GoogLeNet activations (subset of DEEP1B) |
| **SIFT1B (BigANN)** | 1B | 128 | uint8 | L2 | SIFT image features at billion scale |
| **DEEP1B** | 1B | 96 | float32 | L2 | GoogLeNet CNN activations on web images |

SIFT1M and GIST1M are the **in-memory** baselines (compare Vamana vs HNSW/NSG on RAM-sized data). SIFT1B and DEEP1B are the **SSD regime** benchmarks where DiskANN has no competition.

**Standard metric**: `recall@k` — fraction of true $k$-nearest neighbors returned by the approximate search. The paper reports `1-recall@1` (exact nearest neighbor accuracy) and `5-recall@5`.

**Reference**: [ann-benchmarks.com](https://ann-benchmarks.com) hosts reproducible comparisons across methods on these datasets. Larger-scale variants (SPACEV-1B, Turing-ANNS-1B, 100M subsets) appear in follow-up Microsoft Research work.

---

## Why it matters

- **Scale without money**: billion-point search on a $1k workstation instead of a 512 GB RAM server.
- **SSD as first-class citizen**: most ANNS systems treat SSD as a last resort. DiskANN is designed for it from the ground up.
- **Recall–latency Pareto**: DiskANN dominates on the 90–99% recall range, which is the regime real applications care about.

## pgvectorscale & StreamingDiskANN

pgvectorscale es la extensión de Timescale que lleva DiskANN dentro de PostgreSQL. El índice se llama **StreamingDiskANN** — una variante de Vamana con tres diferencias clave respecto al paper original:

| Aspecto | Vamana original | StreamingDiskANN |
|---|---|---|
| Construcción | Batch (requiere todos los puntos) | **Streaming inserts** incrementales sin reconstruir |
| Storage | Sectores raw en SSD | Adaptado al storage engine de Postgres |
| Integración | Stand-alone | Usa el tipo `vector` de pgvector |

**Uso en Postgres:**

```sql
-- Requiere pgvector + pgvectorscale instalados
CREATE INDEX ON embeddings
USING diskann (embedding vector_cosine_ops);

-- Búsqueda: igual que pgvector, el planificador usa el índice
SELECT id, content
FROM embeddings
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

**Jerarquía:**
```
DiskANN paper (Microsoft Research, NeurIPS 2019)
  └── Vamana algorithm
        └── StreamingDiskANN (Timescale)
              └── pgvectorscale (extensión Postgres)
```

La ventaja frente a `pgvector` con HNSW es la misma que DiskANN vs HNSW: menor footprint de RAM a alta escala. pgvector+HNSW mantiene el grafo entero en RAM; pgvectorscale+DiskANN lo mantiene en disco y usa PQ codes en RAM — el mismo trade-off del paper, ahora dentro de SQL.

---

## Used in

[[Dense Retrieval]]
[[Deep Research is a Retrieval Problem]]
[[RAG on 100M documents]]
