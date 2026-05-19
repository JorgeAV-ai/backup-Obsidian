> [!quote] Information
> * calendar Date 2019
> * paper Paper [Link](https://arxiv.org/abs/2003.07820)
> * ? Description:
>   TREC Deep Learning Track 2019 benchmark for ad hoc ranking

## Overview

TREC DL19 (TREC Deep Learning Track 2019) is the first edition of the Deep Learning Track at the Text REtrieval Conference, organized by Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. The track was designed to study ad hoc ranking in the large training data regime, providing large human-labeled training sets based on the MS MARCO dataset.

The track features two tasks: passage ranking and document ranking. Both tasks share the same set of test queries and use large-scale training data with human relevance assessments. The passage ranking task uses a corpus of 8.8 million passages, while the document ranking task uses a corpus of 3.2 million documents. Each task also has two subtasks: full ranking (rank from the entire corpus) and re-ranking (re-rank a provided candidate set).

In total, 15 groups submitted 75 runs using various combinations of deep learning, transfer learning, and traditional IR ranking methods. Results showed that deep learning runs significantly outperformed traditional IR methods, establishing that neural approaches are highly effective when sufficient training data is available.

## Statistics

- **Test queries:** 43
- **Passage corpus:** 8,841,823 passages
- **Document corpus:** 3,213,835 documents
- **Training queries (passage):** ~503,000
- **Training queries (document):** 367,013
- **Tasks:** Passage ranking, Document ranking
- **Subtasks:** Full ranking, Re-ranking
- **Evaluation metrics:** NDCG@10 (primary), MAP, MRR
- **Submitted runs:** 75 (from 15 groups)

## Used in

[[Hypothetical Document Embeddings (HyDE)]]
