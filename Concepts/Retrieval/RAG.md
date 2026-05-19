Retrieval-Augmented Generation (RAG) retrieves relevant documents from an external knowledge base and feeds them into a language model's context, allowing the model to generate answers grounded in up-to-date or private information without modifying its weights.

## What is it?

Large language models have a fixed knowledge cutoff — they only know what was in their training data. They also hallucinate: they confidently generate plausible-sounding but factually incorrect answers. RAG (Lewis et al., 2020) addresses both problems by adding a **retrieval step** before generation. Instead of relying solely on parametric memory (the model's weights), RAG augments the input with relevant documents retrieved from an external corpus.

The core idea is simple: given a user query, (1) find the most relevant documents, (2) concatenate them with the query, and (3) let the LLM generate an answer conditioned on both. This means the model can reference information it was never trained on, and its answers are grounded in actual source documents that can be cited and verified.

**RAG vs. fine-tuning:**

| Aspect | RAG | Fine-tuning |
|---|---|---|
| Model weights | Unchanged | Modified |
| Knowledge source | External, updatable | Baked into weights |
| Update frequency | Instant (update the index) | Requires retraining |
| Cost | Retrieval infrastructure | GPU hours for training |
| Hallucination | Reduced (grounded in docs) | Can still hallucinate |
| Best for | Factual QA, dynamic knowledge | Style, behavior, reasoning |

## How it works

![[basics_rag.png]]

[🔗 Open interactive RAG Demo](../../interactive/rag.html)

### The RAG pipeline

```
Query
  │
  ▼
┌─────────────┐
│  Embedding   │  Encode query to dense vector
│   Model      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Vector     │  Find top-K most similar documents
│  Database    │  (FAISS, Pinecone, Weaviate, Chroma)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Concat      │  prompt = retrieved_docs + query
│  Context     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    LLM       │  Generate answer grounded in context
│  Generation  │
└─────────────┘
```

### Pseudocode: basic RAG pipeline

```python
def rag_pipeline(query, corpus_index, embedding_model, llm, k=5):
    # Step 1: Encode the query
    query_embedding = embedding_model.encode(query)

    # Step 2: Retrieve top-K relevant documents
    top_k_docs = corpus_index.search(query_embedding, k=k)

    # Step 3: Build the augmented prompt
    context = "\n\n".join([doc.text for doc in top_k_docs])
    prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {query}

Answer:"""

    # Step 4: Generate
    answer = llm.generate(prompt)
    return answer, top_k_docs   # return docs for citation
```

### Chunking strategy

Documents are rarely fed whole into the vector database. They are first split into **chunks** — passages of a fixed token length with optional overlap:

```python
def chunk_document(text, chunk_size=512, overlap=64):
    """Split document into overlapping chunks."""
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = detokenize(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap    # slide with overlap
    return chunks
```

**Chunk size trade-off:**

| Chunk size | Retrieval precision | Context richness | Embedding quality |
|---|---|---|---|
| Small (128-256 tokens) | High — focused matches | Low — may lack context | Good — clear signal |
| Medium (512 tokens) | Balanced | Balanced | Good default |
| Large (1024+ tokens) | Low — diluted signal | High — more context | Noisier embedding |

A chunk that is too small may miss surrounding context; a chunk that is too large dilutes the embedding with irrelevant text and wastes the LLM's context window. 512 tokens with 10-20% overlap is a common starting point.

### Vector databases

The retrieval step relies on approximate nearest neighbor (ANN) search over document embeddings:

- **FAISS** (Meta) — open-source, runs locally, supports billions of vectors with IVF/HNSW indices.
- **Pinecone** — managed cloud service, serverless scaling, metadata filtering.
- **Weaviate** — open-source, hybrid search (dense + keyword), GraphQL API.
- **Chroma** — lightweight, designed for RAG prototyping, runs in-process.

## Why it matters

RAG is the most practical way to give LLMs access to private, domain-specific, or frequently changing information. Enterprise knowledge bases, legal documents, medical records, internal wikis — none of these are in the model's training data, and fine-tuning on them is expensive, slow, and risks overfitting. RAG keeps the model general-purpose while dynamically grounding it in relevant source material. It also provides **attribution**: because the retrieved documents are known, the system can cite its sources, which is critical for trust in production applications.

## Used in

- [[Dense Retrieval]]
- [[Hypothetical Document Embeddings (HyDE)|HyDE]]
- [[Embeddings]]
