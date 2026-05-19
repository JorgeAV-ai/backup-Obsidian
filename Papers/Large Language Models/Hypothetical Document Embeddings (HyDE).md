Zero-shot dense retrieval using LLM-generated hypothetical documents as query expansion.

> [!quote] Information
> * @ Conference ACL'23
> * paper Paper [Link](https://arxiv.org/pdf/2212.10496)
> * git Github [Link](https://github.com/texttron/hyde)
> * hf Huggingface [Link](https://huggingface.co/papers/2212.10496)
> *  tag Tags
> 	[[Dense Retrieval]]
> 	[[Instructions-Following Language Models]]
> 	[[Zero-Shot Dense Retrieval]]
> 	[[Generative Retrieval]]
> * calendar Date 20 December 2022
> * ? Motivation:
> 	Hard to create Fully zero-shot Dense Retrieval System
> *  Dataset Datasets:
> 	[[TREC DL19]]
> 	[[TREC DL20]]
> 	[[BEIR]]
> 	[[Mr.TyDi]]
>

### 1. Introduction
#### 1.1 Background
Dense retrieval are widely used for different tasks, such as web search, question answering, fact verification... Allowing the leverage of semantic embedding similarities with great results. Despite its effectiveness, creating dense retrievals that performs well in zero-shot scenarios is not an easy task due to the lack of relevant labels, since, the data will not labeled. The key insight behind this work is that LLMs can generate hypothetical documents that capture relevance patterns even if those documents contain errors or are not accurate, and that a contrastive encoder can effectively encode document similarity in dense retrieval scenarios, filtering out irrelevant details from the generated text.
#### 1.2 Objectives
The primary objective of this research is to develop a zero-shot dense retrieval system that does not require any relevance supervision, being capable of working out-of-the-box and  capable to generalize across different tasks.
#### 1.3 What's New
- First work to use a generative LLM for zero-shot dense retrieval without any relevance supervision.
- Introduces the concept of a hypothetical document as a form of query expansion for dense retrieval.
- Demonstrates that no relevance labels or fine-tuning on target tasks are needed to achieve competitive performance.
### 2. Methodology
#### 2.1 Data
* [[TREC DL19]] & [[TREC DL20]]: Web search queries
* [[BEIR]]: Six low-resource datasets for various retrieval tasks.
* [[Mr.Tydi]]: non-English retrieval tasks in Swahili, Korean, Japanese and Bengali
#### 2.2 Model Architecture
![[Hyde_structure.png]]
Composition:
* **Generative model**: instruction-following model, that generates a hypothetical document based on the query, capturing the relevance patterns.
* **Contrastive encoder:** encodes the hypothetical document into an embedding vector and retrieves similar real documents from the corpus based on vector similarity.

> [!info]- Comments
> *[[Dense Retrieval]]*: Instead of matching keywords (like BM25), both query and document are encoded into dense vectors, and retrieval is done via vector similarity (cosine, dot product). The challenge is that query and document live in different semantic spaces.
>
> *[[Contrastive Learning]]*: The encoder is trained to bring matching query-document pairs closer and push non-matching pairs apart in embedding space. This is how Contriever learns its representations.
#### 2.3 Implementation Details
- **InstructGPT**: Used as the generative model to create hypothetical documents.
- **Contriever**: Used as the contrastive encoder to generate embedding vectors from the hypothetical documents.
- **Pyserini Toolkit**: Used to conduct retrieval experiments.

From what I see in the source code, they use the Gen model to obtain different answers long enough to, convert them to an embedding and then finally with the retriever, obtain responses from different real documents.  
### 3. Results

#### 3.1 Web Search

  ##### 3.1.1 DL19/DL20

|                             | Method   | DL19 map | DL19 ndcg@10 | DL19 recall@1k | DL20 map | DL20 ndcg@10 | DL20 recall@1k |
| --------------------------- | -------- | -------- | ------------ | -------------- | -------- | ------------ | -------------- |
| **w/o relevance judgement** |          |          |              |                |          |              |                |
| BM25                        | 30.1     | 50.6     | 75.0         | 28.6           | 48.0     | 78.6         |                |
| Contriever                  | 24.0     | 44.5     | 74.6         | 24.0           | 42.1     | 75.4         |                |
| HyDE                        | **41.8** | **61.3** | **88.0**     | **38.2**       | **57.9** | 84.4         |                |
| **w/ relevance judgement**  |          |          |              |                |          |              |                |
| DPR                         | 36.5     | 62.2     | 76.9         | 41.8           | **65.3** | 81.4         |                |
| ANCE                        | 37.1     | **64.5** | 75.5         | 40.8           | 64.6     | 77.6         |                |
| Contriever<sup>FT</sup>     | 41.7     | 62.1     | 83.6         | **43.6**       | 63.2     | **85.8**     |                |

  #### 3.1.2 BEIR

| Method nDCG@10              | Scifact  | Arguana  | Trec-Covid | FiQA     | DBPedia  | TREC-NEWS |
| --------------------------- | -------- | -------- | ---------- | -------- | -------- | --------- |
| **w/o relevance judgement** |          |          |            |          |          |           |
| BM25                        | 67.9     | 39.7     | **59.5**   | 23.6     | 31.8     | 39.5      |
| Contriever                  | 64.9     | 37.9     | 27.3       | 24.5     | 29.2     | 34.8      |
| HyDE                        | **69.1** | **46.6** | 59.3       | **27.3** | **36.8** | **44.0**  |
| **w/ relevance judgement**  |          |          |            |          |          |           |
| DPR                         | 31.8     | 17.5     | 33.2       | 29.5     | 26.3     | 16.1      |
| ANCE                        | 50.7     | 41.5     | **65.4**   | 30.0     | 28.1     | 38.2      |
| Contriever<sup>FT</sup>     | 67.7     | 44.6     | 59.6       | **32.9** | **41.3** | 42.8      |

| Method Recall@100           | Scifact  | Arguana  | Trec-Covid | FiQA     | DBPedia  | TREC-NEWS |
| --------------------------- | -------- | -------- | ---------- | -------- | -------- | --------- |
| **w/o relevance judgement** |          |          |            |          |          |           |
| BM25                        | 92.5     | 93.2     | **49.8**   | 54.0     | 46.8     | 44.7      |
| Contriever                  | 92.6     | 90.1     | 17.2       | 56.2     | 45.3     | 42.3      |
| HyDE                        | **96.4** | **97.9** | 41.4       | **62.1** | **47.2** | **50.9**  |
| **w/ relevance judgement**  |          |          |            |          |          |           |
| DPR                         | 72.7     | 75.1     | 21.2       | 34.2     | 34.9     | 21.5      |
| ANCE                        | 81.6     | 93.7     | **45.7**   | 58.1     | 31.9     | 39.8      |
| Contriever<sup>FT</sup>     | 94.7     | 97.7     | 40.7       | **65.6** | **54.1** | **49.2**  |

#### 3.2 Low Resource Retrieval (Mr.TyDi)

| Method                      | Swahili  | Korean   | Japanese | Bengali  |
| --------------------------- | -------- | -------- | -------- | -------- |
| **w/o relevance judgement** |          |          |          |          |
| BM25                        | 38.9     | 28.5     | 21.2     | **41.8** |
| mContriever                 | 38.3     | 22.3     | 19.5     | 35.3     |
| HyDE                        | **41.7** | **30.6** | **30.7** | 41.3     |
| **w/ relevance judgement**  |          |          |          |          |
| mDPR                        | 7.3      | 21.9     | 18.1     | 25.8     |
| mBERT                       | 37.4     | 28.1     | 27.1     | 35.1     |
| XLM-R                       | 35.1     | 32.2     | 24.8     | 41.7     |
| mContriever<sup>FT</sup>    | **51.2** | **34.2** | **32.4** | **42.3** |

#### 3.3 Limitations
- Depends on the quality of the generative model — if the LLM generates poor hypothetical documents, retrieval degrades.
- Higher inference latency than direct embedding-based retrieval: requires an LLM generation step before encoding.
- Underperforms supervised methods (DPR, ANCE, Contriever-FT) when relevance labels ARE available — HyDE is specifically a zero-shot solution.
- Hypothetical documents may contain factual errors, though the contrastive encoder is designed to filter these out.
- On TREC-COVID (domain-specific scientific retrieval), HyDE slightly underperforms BM25, suggesting limits in highly specialized domains.

### 4. Appendix

	Web Search
		Please write a passage to answer the question
		Question: [QUESTION]
		Passage:
	
	SciFact
		Please write a scientific paper passage to support/refute the claim
		Claim: [Claim]
		Passage:
	
	Arguana
		Please write a paper passage to answer the question
		Passage: [PASSAGE]
		Counter Argument:
	
	TREC-COVID
		Please write a scientific paper passage to answer the question
		Question: [QUESTION]
		Passage:
	
	FiQA
		Please write a financial article passage to answer the question
		Question:[QUESTION]
		Passage:
	
	DBPedia-Entity
		Please write a passage to answer the question.
		Question: [QUESTION]
		Passage:
		
	TREC-NEWS
		Please write a news passage about the topic.
		Topic: [TOPIC]
		Passage:
	
	Mr.TyDi
		Please write a passage in Swahili/Korean/Japanese/Bengali to answer the question in detail
		Question: [QUESTION]
		Passage:

### 5. Connections
- Builds on [[Contriever]] as the contrastive encoder for mapping hypothetical documents to dense vectors.
- Uses [[InstructGPT]] as the generative model for producing hypothetical documents from queries.
- Evaluated on [[TREC DL19]], [[TREC DL20]], [[BEIR]], and [[Mr.TyDi]] benchmarks.
- Related to query expansion techniques in traditional IR, but operates in dense embedding space rather than lexical space.
- Follow-up work: RAG approaches like [[Retrieval Augmented Generation (RAG)]] take a different direction — retrieve then generate, vs HyDE's generate then retrieve.
