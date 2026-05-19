# Vault Restructure Design

## Goal

Restructure the Obsidian vault so it scales to many more ML/AI paper notes while remaining fast to review from either a paper-first or concept-first path.

The restructure should improve daily navigation in Obsidian without turning the vault into a rigid taxonomy. The vault should keep readable folder names and preserve the author's existing paper note style.

## Scope

This design covers:

- Moving paper notes under a top-level `Papers/` folder while preserving discipline folders.
- Renaming `Basics/` to `Concepts/`.
- Grouping concept notes into thematic subfolders.
- Adding `Maps/` notes for fast navigation and review.
- Updating repository guidance files to describe the new structure.
- Verifying wikilinks and embeds after migration.

This design does not cover:

- Rewriting paper note content.
- Renaming individual note titles.
- Restructuring `Blog-research/`, `interactive/`, `docs/`, `.obsidian/`, `.superpowers/`, `media/`, `misc/`, or generated root images.
- Changing the paper note template beyond references to the new folders and maps.

## Target Structure

```text
Papers/
  Computer Vision/
  Natural Language Processing/
  Large Language Models/

Concepts/
  Attention/
  Architecture/
  Training/
  Retrieval/
  Compression/
  Math/

Datasets/

Maps/
  Vault Map.md
  Computer Vision.md
  Natural Language Processing.md
  Large Language Models.md
  Attention.md
  Retrieval.md
  Document AI.md
  Efficient Transformers.md
```

## Paper Notes

Existing paper discipline folders move under `Papers/`:

```text
Computer Vision/ -> Papers/Computer Vision/
Natural Language Processing/ -> Papers/Natural Language Processing/
Large Language Models/ -> Papers/Large Language Models/
```

Paper filenames stay unchanged during this phase. This keeps the migration focused on structure and avoids mixing folder moves with semantic title cleanup.

Papers continue to use the established template:

- TL;DR as the first line.
- Strict `Information` callout field order.
- `###` main sections and `####` subsections.
- Detailed `2.2 Model Architecture`.
- Results tables with bold values for the paper's own model.
- `5. Connections` for relationship mapping.

## Concepts

`Basics/` becomes `Concepts/` because many notes are not basic introductory notes anymore. They are reusable conceptual hubs for papers, datasets, and writing.

Initial concept grouping:

```text
Concepts/Attention/
  Self-Attention.md
  Cross-Attention.md
  Flash Attention.md
  KV-Cache.md
  RoPE.md
  Swin Windows.md

Concepts/Training/
  Adam.md
  Backpropagation.md
  BatchNorm.md
  Dropout.md
  Gradient Checkpointing.md
  Layer Normalization.md
  Learning Rate Scheduling.md
  Mixed Precision (FP16 BF16).md
  RLHF.md

Concepts/Architecture/
  Transformer.md
  Residual Connections.md
  Convolution.md
  Depthwise Convolution.md
  ECA.md
  MoE.md

Concepts/Retrieval/
  RAG.md
  Dense Retrieval.md
  DiskANN.md
  Embeddings.md

Concepts/Compression/
  Quantization.md
  LoRA.md

Concepts/Math/
  Softmax.md
  GELU.md
  ReLU.md
  GeGLU.md
  CKA.md
  Contrastive Learning.md
  Tokenization (BPE).md
  Autoregressive Generation.md
  Beam Search.md
  Unpadding.md
```

If a concept clearly fits more than one theme, keep the note in the most useful primary folder and connect it from maps instead of duplicating it.

Do not create empty concept folders in the first migration. Add future folders such as `Optimization/` or `Systems/` only when there are notes that clearly belong there.

## Datasets

`Datasets/` stays as a top-level folder. Dataset notes are stable reference nodes used by papers across disciplines, so moving them under one discipline would make cross-domain reuse worse.

Dataset notes continue to use the existing template:

```markdown
> [!quote] Information
> * calendar Date [YYYY]
> * paper Paper [Link](url)
> * ? Description:
>   [One line]

## Overview
## Statistics
## Used in
```

## Maps

`Maps/` provides fast review entry points. Maps are navigation notes, not long summaries. They should group links and explain relationships in one line when helpful.

`Maps/Vault Map.md` is the main entry point:

```markdown
# Vault Map

## Papers
- [[Computer Vision]]
- [[Natural Language Processing]]
- [[Large Language Models]]

## Core Concepts
- [[Attention]]
- [[Retrieval]]
- [[Document AI]]
- [[Efficient Transformers]]

## Datasets
- [[CORD]]
- [[DocVQA]]
- [[RVL-CDIP]]
- [[TREC DL19]]
- [[TREC DL20]]

## Active Writing
- [[Applications of Reinforcement Learning in Current Frontier LLM Models]]
- [[Quantizating LLMs, How it works and what you should be concerned about]]
- [[RAG on 100M documents]]
```

Domain maps use this shape:

```markdown
# Computer Vision

## Papers
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]

## Concepts
- [[Self-Attention]]
- [[Convolution]]

## Datasets
- [[ImageNet]]
- [[MS COCO]]

## Threads
- Efficient high-resolution vision
- OCR-free document understanding
- Windowed and hierarchical attention
```

Theme maps use this shape:

```markdown
# Attention

## Core Concepts
- [[Self-Attention]]
- [[Cross-Attention]]

## Papers Using It
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]] - carrier tokens plus hierarchical attention.

## Related Maps
- [[Efficient Transformers]]
- [[Large Language Models]]
- [[Computer Vision]]
```

## Linking Rules

- Papers link to concepts and datasets from `2.1 Data`, `2.2 Model Architecture`, `> [!info]- Comments`, and `5. Connections`.
- Concepts link back to papers in `## Used in`.
- Datasets link back to papers in `## Used in`.
- Maps do not duplicate explanations from notes. They provide orientation and curated paths.
- Wikilinks should remain basename-based and readable. Folder paths should only be used inside links if Obsidian reports ambiguous note names.

## Migration Rules

The implementation should be conservative:

1. Create `Papers/`, `Concepts/`, and `Maps/`.
2. Move discipline paper folders under `Papers/`.
3. Rename `Basics/` to `Concepts/`.
4. Move concept notes into the approved thematic subfolders.
5. Add the initial map notes.
6. Update `AGENTS.md` and `CLAUDE.md`.
7. Verify wikilinks and image embeds.

Do not move unrelated generated assets or writing work in this phase.

## Verification

After migration, run checks for:

- Broken `[[wikilinks]]`.
- Broken `![[embeds]]`.
- Duplicate note basenames that could make Obsidian resolution ambiguous.
- References to the old `Basics/` folder in guidance docs.
- Any accidental moves under `Blog-research/`, `interactive/`, `docs/`, `media/`, or `misc/`.

The design commit and migration commit should be separate.
