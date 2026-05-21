# Obsidian Canvas Research Hub Design

## Goal

Create one Obsidian Canvas that acts as a personal research control room for paper reading, concept review, active writing, and learning direction.

The Canvas should not replace Markdown notes. It should help decide what to read, connect, revisit, and work on next.

## Core Principle

```text
Markdown = content
Maps = indexes
Canvas = direction and priority
```

Long explanations stay in Markdown notes under `Papers/`, `Concepts/`, `Datasets/`, `Maps/`, and `Blog-research/`.

The Canvas stores lightweight visual state, priority, and relationships.

## Target File

Create one initial Canvas:

```text
Canvas/Personal AI Research Hub.canvas
```

Do not create multiple canvases in the first version. If the main Canvas becomes too large, split it later into domain-specific canvases.

## Layout

The Canvas has two main zones.

### Operative Layer

The top area is for daily work:

```text
Inbox
Reading Pipeline
Active Projects
Review Queue
```

This zone answers:

- What should I process next?
- What am I reading now?
- What needs summarizing?
- What projects are active?
- What concepts should I revisit?

### Knowledge Layer

The bottom area is for visual orientation:

```text
Domains
Concept Clusters
Paper-to-Concept Connections
Learning Paths
```

This zone answers:

- How do papers connect to concepts?
- Which domains am I building?
- Which learning path makes sense next?
- Which concept clusters are central?

## Operative Cards

### Inbox

Create lightweight text cards:

```text
New paper to process
New concept to learn
Question to investigate
Potential blog idea
```

Inbox is temporary. Items should move out during review.

### Reading Pipeline

Create visual columns:

```text
To Read -> Reading -> Summarizing -> Connected -> Revisit
```

Rules:

- A paper appears in only one pipeline state at a time.
- `Connected` means the paper has meaningful links to concepts, datasets, or maps.
- `Revisit` means useful but not currently active.

Seed the pipeline with current paper notes:

- `[[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]`
- `[[Swin Transformer, Hierarchical Vision Transformer using Shifted Windows]]`
- `[[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]`
- `[[OCR-free Document Understanding Transformer (Donut)]]`
- `[[Visual Document Understanding (VDU)]]`
- `[[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]`
- `[[Hypothetical Document Embeddings (HyDE)]]`

### Active Projects

Show a small number of active writing or production items. The first version should include:

- `[[Applications of Reinforcement Learning in Current Frontier LLM Models]]`
- `[[Quantizating LLMs, How it works and what you should be concerned about]]`
- `[[RAG on 100M documents]]`

Keep no more than three active projects visible. Move inactive work to `Blog-research/` or a map instead of keeping it as prominent Canvas clutter.

### Review Queue

Seed with high-value concepts:

- `[[Self-Attention]]`
- `[[Flash Attention]]`
- `[[RoPE]]`
- `[[Dense Retrieval]]`
- `[[Quantization]]`
- `[[RLHF]]`
- `[[BatchNorm]]`

Concepts do not need workflow states. They can be grouped visually as:

```text
Core
Weak
Review soon
```

## Knowledge Cards

### Domains

Add domain nodes linked to existing maps:

- `[[Computer Vision]]`
- `[[Natural Language Processing]]`
- `[[Large Language Models]]`
- `[[Document AI]]`

### Concept Clusters

Add major concept cluster nodes linked to maps or concept notes:

- `[[Attention]]`
- `[[Retrieval]]`
- `[[Efficient Transformers]]`
- `[[Document AI]]`
- `[[Quantization]]`
- `[[LoRA]]`
- `[[Training]]`
- `[[Architecture]]`

If a cluster does not yet have a map note, it can still exist as a Canvas text card. Do not create placeholder Markdown notes only to satisfy the Canvas.

### Recommended Connections

Draw only high-signal connections:

```text
Attention -> Efficient Transformers
Retrieval -> Large Language Models
Document AI -> Computer Vision
Quantization -> Efficient Transformers
LoRA -> Efficient Inference
ModernBERT -> Attention
HyDE -> Retrieval
FasterViT -> Efficient Transformers
Donut -> Document AI
```

The Canvas should not replicate the full Obsidian graph. It should show relationships that help choose the next reading or review action.

## Maintenance Rules

1. If a Canvas card needs more than 3-5 lines, convert it into a Markdown note and link it.
2. Keep each paper in one pipeline state.
3. Keep active projects capped at three visible items.
4. Review the Canvas once per week for about ten minutes.
5. During weekly review:
   - Empty Inbox.
   - Move papers through the pipeline.
   - Pick 1-3 concepts for review.
   - Remove stale cards.
   - Add only useful new connections.
6. Do not add YAML, Dataview, or heavy metadata in the first version.

## Out Of Scope

This design does not include:

- Dataview dashboards.
- YAML status fields on every paper.
- Multiple domain canvases.
- Automated Canvas generation.
- Rewriting existing paper or concept notes.
- Changing the vault structure again.

## Verification

After implementation:

- Confirm `Canvas/Personal AI Research Hub.canvas` exists.
- Confirm it links to existing notes where possible.
- Confirm no placeholder Markdown notes were created just for Canvas nodes.
- Confirm `Papers/`, `Concepts/`, `Datasets/`, `Maps/`, and `Blog-research/` remain the source of truth.
- Confirm the Canvas has both operative and knowledge zones.
