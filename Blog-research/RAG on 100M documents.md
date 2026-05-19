Search at 100M documents — what breaks, what you fix, and what you learn.

> [!quote] Source
> * paper Blog Post [hornet.dev/blog/100m-doc-search-part-1-what-we-learned](https://hornet.dev/blog/100m-doc-search-part-1-what-we-learned)
> * calendar Date [2025]
> * ? Summary:
>   Engineering lessons from building a 100M-document search engine. Every experiment takes a full day — so iteration speed becomes the real bottleneck, not recall metrics.

![[rag_100m_diagram.png]]

---

## Problems

### 1. Slow Iteration
Every experiment at 100M documents costs a full day. Full reindexing takes **27–30.5 hours** without checkpointing. Decision cycles stretched to multiple days per configuration test.

> The real constraint is not hardware — it's the feedback loop. At this scale, you can't afford to be wrong.

### 2. Dirty Corpus
Initial retrieval results were "embarrassingly bad" — but the bug was not in the retrieval layer. **Login screens, 404 pages, and thin content** appeared as top results because their embedding vectors happened to be nearest to queries.

### 3. Ingestion Bottleneck
The feed pipeline maxed out at **300 docs/sec** (3–4 days for the full corpus). The team had no baseline to know if this was good or bad — visibility was the first problem.

### 4. Parameter Uncertainty
Expected: one optimal config. Found: **different query types favor different retrieval modes**. Human queries behave differently from agentic queries (longer, more semantically precise). No single setting wins everywhere.

### 5. Preprocessing Cascades
Bad decisions during corpus preparation surfaced as measurable quality failures at evaluation time. Short documents, near-duplicates, and noise **wasted embedding compute** and polluted the index before any query was run.

---

## Solutions

### A. Corpus Cleaning
Applied **minimum content-length filters** to remove noise before embedding. Quality improved dramatically. The retrieval system wasn't broken — the input was.

### B. Ingestion Optimization
Tuned batching logic, increased thread counts, and ran within **AWS to minimize network latency**. Result: **5× throughput improvement** — from 300 to ~1,500 docs/sec.

### C. Hybrid Retrieval Testing
Tested both:
- **Pure ANN** (approximate nearest neighbor — vector similarity only)
- **Hybrid** (ANN + keyword matching)

Evaluated across multiple query datasets including the [[MIMICS]] dataset.

### D. 4-Stage Pipeline Architecture
| Stage | Description |
|---|---|
| 1. Corpus Prep | Clean, deduplicate, filter short content |
| 2. Embedding Generation | Encode documents at scale |
| 3. Distributed Ingestion | Kubernetes-based parallel feed |
| 4. Evaluation | Multi-mode retrieval + multi-metric scoring |

---

## Key Insights

> [!important] Iteration Speed > Any Single Parameter
> "Iteration speed matters more than any single parameter." Teams don't lack ideas — they lack the ability to **test them affordably at production scale**.

### Offline Costs Dominate Quality
Query-time metrics (latency, throughput, recall) are not the full picture. The real determinants of how fast you improve quality are:
- **Reindex time** — how long before you can test a change
- **Embedding cost** — how expensive it is to re-encode the corpus
- **Discovering preprocessing mistakes late** — the hidden cost multiplier

### Design for Learning, Not for Optimal Config
The conclusion is architectural: build systems that let you **experiment cheaply and iterate fast**. Early investment in corpus validation and rapid iteration loops yields better long-term results than optimizing a single configuration.

---

## What's Coming (Parts 2 & 3)

| Part | Topic |
|---|---|
| Part 2 | ANN parameter tuning — effects on recall and latency |
| Part 3 | Hybrid search behavior across different query patterns |

---

## Connections

- [[Dense Retrieval]] — the ANN-based retrieval foundation
- [[HyDE]] — generation-augmented retrieval, alternative strategy
- [[DocVQA]] — document understanding at scale, complementary problem
- [[CORD]] — structured document retrieval benchmarks
