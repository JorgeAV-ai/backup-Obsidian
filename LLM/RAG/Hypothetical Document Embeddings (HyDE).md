
> [!quote] Information 
> * paper Paper [Link](https://arxiv.org/pdf/2212.10496)
> *  tag Tags [[Retrieval Augmented Generation (RAG)]], Dense Retrieval
> * calendar Date 20 December 2022
> * ? Motivation: 
> 	Hard to create Fully zero-shot Dense Retrieval System
> *  Dataset Datasets:
> 	[[TREC DL19]] 
> 	[[TREC DL20]]
> 	[[BEIR]]
> 	[[Mr.Tydi dataset]]
> * Fields Related fields: 
> 	[[Dense Retrieval]]
> 	[[Instructions-Following Language Models]]
> 	[[Zero-Shot Dense Retrieval]]
> 	[[Generative Retrieval]]
> 

### 1. Introduction

![[Pasted image 20240721200934.png]]
#### 1.1 Background:
The lack of relevance labels is one of the challengues discussed in this paper, where it state that is hard to train the data without the possibility to indicate where the relevant information is.  
#### 1.2 Objectives:

### 2. Methodology
#### 2.1 Assumptions
#### 2.2 Data
#### 2.3 Model Architecture
#### 2.4 Implementation Details

### 3. Results

### 4. Discussion & Conclusion

### 5. Appendix

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
	
### 5. Code
