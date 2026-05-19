> [!quote] Information
> * calendar Date 2020
> * paper Paper [Link](https://arxiv.org/abs/2102.07662)
> * ? Description:
>   TREC Deep Learning Track 2020 benchmark for ad hoc ranking

## Overview

TREC DL20 (TREC Deep Learning Track 2020) is the second edition of the Deep Learning Track at the Text REtrieval Conference, organized by Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. Like its predecessor, the track studies ad hoc ranking in the large training data regime, using the MS MARCO dataset as the basis for training labels and corpora.

The track continues with the same two tasks as the 2019 edition: passage ranking and document ranking, each with full ranking and re-ranking subtasks. The 2020 edition uses the same corpora as DL19 but with a new and larger set of test queries (200 queries compared to 43 in DL19). Documents were judged on a four-point relevance scale: Not Relevant (0), Relevant (1), Highly Relevant (2), and Perfect (3).

A notable addition in the 2020 track was the release of the ORCAS dataset, a large-scale click dataset constructed from search engine logs, which participants could use as supplementary training data. Results further confirmed that rankers with BERT-style pretraining significantly outperform other approaches in the large data regime.

## Statistics

- **Test queries:** 200
- **Passage corpus:** 8,841,823 passages
- **Document corpus:** 3,213,835 documents
- **Training queries (document):** 367,013
- **Training queries (passage):** ~532,761 relevance labels
- **Tasks:** Passage ranking, Document ranking
- **Subtasks:** Full ranking, Re-ranking
- **Relevance scale:** 4-point (0-3)
- **Evaluation metrics:** NDCG@10 (primary), MAP, MRR

## Used in

[[Hypothetical Document Embeddings (HyDE)]]
