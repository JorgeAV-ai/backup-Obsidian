Dense retrieval encodes queries and documents as dense vectors, replacing keyword matching with learned semantic similarity for information retrieval.

## What is it?

Dense retrieval is an information retrieval paradigm that represents both queries and documents as continuous, high-dimensional vectors (embeddings) produced by neural networks. Instead of relying on exact keyword overlap — as traditional methods like BM25 or TF-IDF do — dense retrieval captures semantic meaning, so a query like "how to fix a flat tire" can match a document titled "changing a punctured wheel" even though they share no words.

The standard architecture is the **bi-encoder**: two separate (or shared-weight) encoder networks, one for queries ($E_q$) and one for documents ($E_d$). At indexing time, every document in the corpus is encoded once and stored in a vector index. At query time, only the query needs to be encoded, and retrieval reduces to a nearest-neighbor search in embedding space. This separation is what makes dense retrieval practical at scale — you never re-encode the corpus.

Compared to **sparse retrieval** (BM25, TF-IDF), which builds inverted indices over discrete tokens and scores documents by term-frequency statistics, dense retrieval trades interpretability and zero-shot robustness for the ability to learn task-specific notions of relevance. In practice, hybrid systems that combine both signals often outperform either alone.

## How it works

![[basics_dense_retrieval.png]]

[🔗 Open interactive Retrieval Explorer](../../interactive/retrieval.html)

**Relevance scoring:**

$$\text{score}(q, d) = \text{sim}\!\bigl(E_q(q),\; E_d(d)\bigr)$$

where $\text{sim}$ is typically cosine similarity or dot product.

**Bi-encoder retrieval pipeline (pseudocode):**

```
# === Offline indexing ===
for d in corpus:
    v_d = E_d(d)            # encode document to dense vector
    index.add(v_d, id=d)    # add to FAISS / ANN index

# === Online retrieval ===
v_q = E_q(query)             # encode query
top_k = index.search(v_q, k) # approximate nearest neighbor search
return top_k
```

**Step-by-step:**

1. **Encode documents** — pass each document through $E_d$ to obtain a fixed-size vector (e.g., 768-d from a BERT-based encoder). Store all vectors in an index.
2. **Build an ANN index** — use libraries like **FAISS** (Facebook AI Similarity Search) to construct an approximate nearest neighbor (ANN) structure (e.g., IVF, HNSW) that enables sub-linear search over millions of vectors.
3. **Encode the query** — at inference, pass the user query through $E_q$ to get its embedding.
4. **Search** — retrieve the top-$k$ nearest document vectors from the ANN index. Return the corresponding documents.

**Sparse vs. Dense comparison:**

| Aspect | Sparse (BM25 / TF-IDF) | Dense Retrieval |
|---|---|---|
| Representation | Sparse token counts | Learned dense vectors |
| Matching | Exact keyword overlap | Semantic similarity |
| Index structure | Inverted index | ANN index (FAISS, ScaNN) |
| Training | Unsupervised (statistical) | Supervised / contrastive |
| Weakness | Vocabulary mismatch | Needs training data; domain shift |

## Why it matters

Keyword-based retrieval fails when the user's vocabulary differs from the document's vocabulary — the so-called *vocabulary mismatch* problem. Dense retrieval solves this by learning a shared semantic space where meaning, not surface form, determines relevance. This is critical for open-domain question answering, retrieval-augmented generation (RAG), and any system that needs to find relevant passages across large, heterogeneous corpora.

## Used in

[[Hypothetical Document Embeddings (HyDE)]]
