# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What This Is

An Obsidian vault containing structured notes on ML/AI research papers. The author (Jorge) uses it as a personal knowledge base to track papers across Computer Vision, NLP, LLMs, and Datasets. It is backed by git with periodic "vault backup" commits.

## Repository Structure

- **Papers/Computer Vision/**: Paper notes on vision models (FasterViT, Donut, Skip-Attention, Swin, VDU)
- **Papers/Natural Language Processing/**: Paper notes on NLP models (ModernBERT)
- **Papers/Large Language Models/**: Paper notes on LLM and retrieval techniques (HyDE)
- **Concepts/**: Reusable concept notes grouped by theme (Attention, Training, Architecture, Retrieval, Compression, Math)
- **Datasets/**: Notes on benchmark datasets referenced by papers (TREC DL19/20, DocVQA, RVL-CDIP, CORD)
- **Maps/**: Navigation notes for paper-first and concept-first review paths
- **Blog-research/**: Drafts and research material for long-form writing
- **interactive/**: Standalone HTML interactives for concept exploration
- **misc/** and **media/**: Images, diagrams, generated figures, and pasted assets referenced in notes

## Paper Note Template (FasterViT is the gold standard)

### Callout — field order is strict:
```
TL;DR one-line summary here (before the callout)

> [!quote] Information
> * @ Conference [NAME]
> * paper Paper [Link](url)
> * git Github [Link](url)
> * hf Huggingface [Link](url)
> *  tag Tags
> 	[[Tag1]]
> 	[[Tag2]]
> * calendar Date [DD Month YYYY]
> * ? Motivation:
> 		[1-3 lines max]
> *  Dataset Datasets:
> 	[[Dataset1]]
```

### Sections — heading levels are `###` (main) / `####` (sub):
```
### 1. Introduction
#### 1.1 Background        ← SHORT (1-3 lines). Core problem + key assumptions merged in.
#### 1.2 Objectives        ← SHORT. What the paper proposes.
#### 1.3 What's New        ← Bullet list: what makes this paper unique vs. everything else.

### 2. Methodology
#### 2.1 Data              ← Datasets with [[wikilinks]], brief description of each.
#### 2.2 Model Architecture  ← THE HEART. Detailed with:
                              - Diagrams: ![[image.png]]
                              - Pseudocode in fenced code blocks (Python-like, simplified)
                              - LaTeX formulas for key equations
                              - > [!info]- Comments   ← collapsible callouts after code
                                with personal explanations of concepts
#### 2.3 Implementation Details  ← Concise bullet list (optimizer, LR, epochs, GPUs, batch size).

### 3. Results             ← Markdown tables grouped by category.
                              Bold the paper's own model results.
                              Minimal commentary — tables speak for themselves.
                              Include #### Limitations subsection when relevant
                              (when NOT to use the model).

### 4. Appendix            ← Optional. Only if there's something worth noting.

### 5. Connections         ← Relationship map with [[wikilinks]] + context:
                              what it builds on, what it's compared against,
                              follow-up work. Creates meaningful graph view edges.
```

### Key style rules:
- TL;DR as first line of file — one sentence to identify the paper without opening it.
- Intros (1.1/1.2) are SHORT. Don't write academic summaries.
- What's New (1.3) answers "what does this do that others don't?"
- No separate Assumptions section — merge into Background.
- Architecture (2.2) gets the most detail — pseudocode + comments are for re-understanding without re-reading the paper.
- `> [!info]- Comments` collapsible callouts after pseudocode serve as mini self-lessons (formulas, concept explanations). The `-` makes them collapsible.
- Results tables use bold (`**value**`) for the paper's model. Include Limitations when the paper has clear failure cases.
- Connections (5) maps relationships — not just tags, but HOW papers relate (builds on, compared against, follow-up).

## Conventions

- Internal cross-references use Obsidian `[[wikilinks]]` (e.g., `[[TREC DL19]]`, `[[Dense Retrieval]]`)
- Images are embedded with `![[filename.png]]` and stored in `misc/`
- Commits follow the pattern: `vault backup: YYYY-MM-DD HH:MM:SS`
- Content is written in English
- Tables use standard Markdown format for benchmark comparisons

## Concept Note Template
```
TL;DR one-line summary

## What is it?
[Intuitive explanation, 2-3 paragraphs. Teach it to yourself.]

## How it works
[Formulas (LaTeX), pseudocode in fenced code blocks, step-by-step. THE HEART.]

## Why it matters
[What problem it solves, what came before it]

## Used in
[[Paper1]]
[[Paper2]]
```

Concept notes are the deep-dive versions of the `> [!info]- Comments` in papers. Comments = quick 2-3 line reminder. Concepts = full explanation with formulas and pseudocode. Papers link to Concepts via `[[wikilinks]]`, creating central hub nodes in the graph view.

## Dataset Note Template
```
> [!quote] Information
> * calendar Date [YYYY]
> * paper Paper [Link](url)
> * ? Description:
>   [One line]

## Overview
[2-3 paragraphs]

## Statistics
- [Key stats as bullet list]

## Used in
[[Paper1]]
```
