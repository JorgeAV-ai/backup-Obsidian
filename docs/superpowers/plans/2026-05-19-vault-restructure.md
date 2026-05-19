# Vault Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Obsidian vault so paper notes, concepts, datasets, and review maps scale cleanly while preserving readable wikilinks.

**Architecture:** Keep permanent knowledge in clear top-level areas: `Papers/`, `Concepts/`, `Datasets/`, and `Maps/`. Move existing notes conservatively, preserve filenames, add map notes as lightweight navigation hubs, and update agent guidance to match the new structure.

**Tech Stack:** Obsidian Markdown, git, POSIX shell commands, `rg`, `find`.

---

## Files And Responsibilities

- Create: `Papers/` as the top-level home for paper notes grouped by discipline.
- Move: `Computer Vision/` to `Papers/Computer Vision/`.
- Move: `Natural Language Processing/` to `Papers/Natural Language Processing/`.
- Move: `Large Language Models/` to `Papers/Large Language Models/`.
- Create: `Concepts/` as the renamed and grouped replacement for `Basics/`.
- Move: `Basics/*.md` into `Concepts/<theme>/`.
- Keep: `Datasets/` as a top-level reusable reference area.
- Create: `Maps/*.md` as navigation notes.
- Modify: `AGENTS.md` to describe the new structure.
- Modify: `CLAUDE.md` to describe the new structure.
- Do not modify or move: `Blog-research/`, `interactive/`, `docs/` except this plan, `.obsidian/`, `.superpowers/`, `media/`, `misc/`, or generated root images.

## Task 1: Baseline Inventory

**Files:**
- Read: repository tree and target folders.
- No file changes.

- [ ] **Step 1: Confirm current branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected:

- Current branch is visible.
- There may be pre-existing modified and untracked files. Do not revert or stage unrelated changes.

- [ ] **Step 2: Capture current permanent note inventory**

Run:

```bash
find Basics 'Computer Vision' 'Natural Language Processing' 'Large Language Models' Datasets -maxdepth 1 -type f -name '*.md' -print | sort
```

Expected output includes:

```text
Basics/Adam.md
Basics/Self-Attention.md
Computer Vision/Fast Vision Transformers with Hierarchical Attention (FasterViT).md
Large Language Models/Hypothetical Document Embeddings (HyDE).md
Natural Language Processing/A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT).md
Datasets/CORD.md
```

- [ ] **Step 3: Check that target folders do not already contain migrated notes**

Run:

```bash
find Papers Concepts Maps -maxdepth 3 -type f -name '*.md' -print 2>/dev/null | sort
```

Expected:

- Either no output, or only files from a previous partial migration.
- If previous partial migration files exist, stop and inspect before moving anything.

## Task 2: Move Paper Folders Under `Papers/`

**Files:**
- Create: `Papers/`
- Move: `Computer Vision/` to `Papers/Computer Vision/`
- Move: `Natural Language Processing/` to `Papers/Natural Language Processing/`
- Move: `Large Language Models/` to `Papers/Large Language Models/`

- [ ] **Step 1: Create `Papers/`**

Run:

```bash
mkdir -p Papers
```

Expected:

- `Papers/` exists.

- [ ] **Step 2: Move paper discipline folders**

Run:

```bash
mv 'Computer Vision' Papers/
mv 'Natural Language Processing' Papers/
mv 'Large Language Models' Papers/
```

Expected:

- The three paper discipline folders now live under `Papers/`.

- [ ] **Step 3: Verify paper notes moved and filenames stayed unchanged**

Run:

```bash
find Papers -maxdepth 2 -type f -name '*.md' -print | sort
```

Expected output includes:

```text
Papers/Computer Vision/Fast Vision Transformers with Hierarchical Attention (FasterViT).md
Papers/Computer Vision/OCR-free Document Understanding Transformer (Donut).md
Papers/Computer Vision/Skip-Attention, Improving Vision Transformers by Paying Less Attention.md
Papers/Computer Vision/Swin Transformer, Hierarchical Vision Transformer using Shifted Windows.md
Papers/Computer Vision/Visual Document Understanding (VDU).md
Papers/Large Language Models/Hypothetical Document Embeddings (HyDE).md
Papers/Natural Language Processing/A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT).md
```

- [ ] **Step 4: Commit paper folder move**

Run:

```bash
git add Papers 'Computer Vision' 'Natural Language Processing' 'Large Language Models'
git commit -m "docs: move paper notes under Papers"
```

Expected:

- Commit succeeds.
- Unrelated dirty files remain unstaged.

## Task 3: Move `Basics/` Notes Into `Concepts/`

**Files:**
- Create: `Concepts/Attention/`
- Create: `Concepts/Training/`
- Create: `Concepts/Architecture/`
- Create: `Concepts/Retrieval/`
- Create: `Concepts/Compression/`
- Create: `Concepts/Math/`
- Move: every `Basics/*.md` file into one concept theme folder.

- [ ] **Step 1: Create concept theme folders**

Run:

```bash
mkdir -p Concepts/Attention Concepts/Training Concepts/Architecture Concepts/Retrieval Concepts/Compression Concepts/Math
```

Expected:

- Six concept theme folders exist.

- [ ] **Step 2: Move attention concepts**

Run:

```bash
mv 'Basics/Self-Attention.md' Concepts/Attention/
mv 'Basics/Cross-Attention.md' Concepts/Attention/
mv 'Basics/Flash Attention.md' Concepts/Attention/
mv 'Basics/KV-Cache.md' Concepts/Attention/
mv 'Basics/RoPE.md' Concepts/Attention/
```

Expected:

- Attention concept notes are under `Concepts/Attention/`.

- [ ] **Step 3: Move training concepts**

Run:

```bash
mv 'Basics/Adam.md' Concepts/Training/
mv 'Basics/Backpropagation.md' Concepts/Training/
mv 'Basics/BatchNorm.md' Concepts/Training/
mv 'Basics/Dropout.md' Concepts/Training/
mv 'Basics/Gradient Checkpointing.md' Concepts/Training/
mv 'Basics/Layer Normalization.md' Concepts/Training/
mv 'Basics/Learning Rate Scheduling.md' Concepts/Training/
mv 'Basics/Mixed Precision (FP16 BF16).md' Concepts/Training/
mv 'Basics/RLHF.md' Concepts/Training/
```

Expected:

- Training concept notes are under `Concepts/Training/`.

- [ ] **Step 4: Move architecture concepts**

Run:

```bash
mv 'Basics/Transformer.md' Concepts/Architecture/
mv 'Basics/Residual Connections.md' Concepts/Architecture/
mv 'Basics/Convolution.md' Concepts/Architecture/
mv 'Basics/Depthwise Convolution.md' Concepts/Architecture/
mv 'Basics/ECA.md' Concepts/Architecture/
mv 'Basics/MoE.md' Concepts/Architecture/
```

Expected:

- Architecture concept notes are under `Concepts/Architecture/`.

- [ ] **Step 5: Move retrieval concepts**

Run:

```bash
mv 'Basics/RAG.md' Concepts/Retrieval/
mv 'Basics/Dense Retrieval.md' Concepts/Retrieval/
mv 'Basics/DiskANN.md' Concepts/Retrieval/
mv 'Basics/Embeddings.md' Concepts/Retrieval/
```

Expected:

- Retrieval concept notes are under `Concepts/Retrieval/`.

- [ ] **Step 6: Move compression concepts**

Run:

```bash
mv 'Basics/Quantization.md' Concepts/Compression/
mv 'Basics/LoRA.md' Concepts/Compression/
```

Expected:

- Compression concept notes are under `Concepts/Compression/`.

- [ ] **Step 7: Move math and generation concepts**

Run:

```bash
mv 'Basics/Softmax.md' Concepts/Math/
mv 'Basics/GELU.md' Concepts/Math/
mv 'Basics/ReLU.md' Concepts/Math/
mv 'Basics/GeGLU.md' Concepts/Math/
mv 'Basics/CKA.md' Concepts/Math/
mv 'Basics/Contrastive Learning.md' Concepts/Math/
mv 'Basics/Tokenization (BPE).md' Concepts/Math/
mv 'Basics/Autoregressive Generation.md' Concepts/Math/
mv 'Basics/Beam Search.md' Concepts/Math/
mv 'Basics/Unpadding.md' Concepts/Math/
```

Expected:

- Math and generation concept notes are under `Concepts/Math/`.

- [ ] **Step 8: Verify `Basics/` is empty or removable**

Run:

```bash
find Basics -maxdepth 1 -type f -name '*.md' -print 2>/dev/null | sort
```

Expected:

- No output.

- [ ] **Step 9: Remove empty `Basics/` folder**

Run:

```bash
rmdir Basics
```

Expected:

- `Basics/` is removed if it is empty.

- [ ] **Step 10: Verify concept note inventory**

Run:

```bash
find Concepts -maxdepth 2 -type f -name '*.md' -print | sort
```

Expected output includes:

```text
Concepts/Attention/Flash Attention.md
Concepts/Attention/Self-Attention.md
Concepts/Architecture/Transformer.md
Concepts/Compression/Quantization.md
Concepts/Retrieval/RAG.md
Concepts/Training/BatchNorm.md
Concepts/Math/Softmax.md
```

- [ ] **Step 11: Commit concept move**

Run:

```bash
git add Concepts
git commit -m "docs: organize basics as concepts"
```

Expected:

- Commit succeeds.
- Unrelated dirty files remain unstaged.

## Task 4: Create Map Notes

**Files:**
- Create: `Maps/Vault Map.md`
- Create: `Maps/Computer Vision.md`
- Create: `Maps/Natural Language Processing.md`
- Create: `Maps/Large Language Models.md`
- Create: `Maps/Attention.md`
- Create: `Maps/Retrieval.md`
- Create: `Maps/Document AI.md`
- Create: `Maps/Efficient Transformers.md`

- [ ] **Step 1: Create `Maps/`**

Run:

```bash
mkdir -p Maps
```

Expected:

- `Maps/` exists.

- [ ] **Step 2: Create `Maps/Vault Map.md`**

Create `Maps/Vault Map.md` with:

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

- [ ] **Step 3: Create `Maps/Computer Vision.md`**

Create `Maps/Computer Vision.md` with:

```markdown
# Computer Vision

## Papers
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]
- [[Swin Transformer, Hierarchical Vision Transformer using Shifted Windows]]
- [[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]
- [[OCR-free Document Understanding Transformer (Donut)]]
- [[Visual Document Understanding (VDU)]]

## Concepts
- [[Self-Attention]]
- [[Flash Attention]]
- [[RoPE]]
- [[Convolution]]
- [[Depthwise Convolution]]
- [[Transformer]]

## Datasets
- [[ImageNet]]
- [[MS COCO]]
- [[ADE20K]]
- [[DocVQA]]
- [[RVL-CDIP]]
- [[CORD]]

## Threads
- Efficient high-resolution vision
- OCR-free document understanding
- Windowed and hierarchical attention
```

- [ ] **Step 4: Create `Maps/Natural Language Processing.md`**

Create `Maps/Natural Language Processing.md` with:

```markdown
# Natural Language Processing

## Papers
- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]

## Concepts
- [[Transformer]]
- [[Self-Attention]]
- [[Flash Attention]]
- [[RoPE]]
- [[Unpadding]]
- [[Embeddings]]

## Datasets
- [[TREC DL19]]
- [[TREC DL20]]

## Threads
- Efficient encoder models
- Long-context bidirectional language modeling
- Retrieval-oriented evaluation
```

- [ ] **Step 5: Create `Maps/Large Language Models.md`**

Create `Maps/Large Language Models.md` with:

```markdown
# Large Language Models

## Papers
- [[Hypothetical Document Embeddings (HyDE)]]

## Concepts
- [[RAG]]
- [[Dense Retrieval]]
- [[Embeddings]]
- [[Transformer]]
- [[Tokenization (BPE)]]
- [[KV-Cache]]
- [[Quantization]]
- [[LoRA]]
- [[RLHF]]

## Datasets
- [[TREC DL19]]
- [[TREC DL20]]

## Threads
- Retrieval-augmented generation
- Efficient inference and adaptation
- Evaluation with retrieval benchmarks
```

- [ ] **Step 6: Create `Maps/Attention.md`**

Create `Maps/Attention.md` with:

```markdown
# Attention

## Core Concepts
- [[Self-Attention]]
- [[Cross-Attention]]
- [[Flash Attention]]
- [[KV-Cache]]
- [[RoPE]]

## Papers Using It
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]] - carrier tokens plus hierarchical attention.
- [[Swin Transformer, Hierarchical Vision Transformer using Shifted Windows]] - shifted windows for local and cross-window exchange.
- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]] - efficient long-context encoder attention.

## Related Maps
- [[Efficient Transformers]]
- [[Large Language Models]]
- [[Computer Vision]]
```

- [ ] **Step 7: Create `Maps/Retrieval.md`**

Create `Maps/Retrieval.md` with:

```markdown
# Retrieval

## Core Concepts
- [[RAG]]
- [[Dense Retrieval]]
- [[DiskANN]]
- [[Embeddings]]

## Papers Using It
- [[Hypothetical Document Embeddings (HyDE)]] - generates hypothetical documents to improve dense retrieval.
- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]] - evaluated on retrieval-oriented benchmarks.

## Datasets
- [[TREC DL19]]
- [[TREC DL20]]

## Related Maps
- [[Large Language Models]]
- [[Natural Language Processing]]
```

- [ ] **Step 8: Create `Maps/Document AI.md`**

Create `Maps/Document AI.md` with:

```markdown
# Document AI

## Papers
- [[OCR-free Document Understanding Transformer (Donut)]]
- [[Visual Document Understanding (VDU)]]

## Concepts
- [[Transformer]]
- [[Self-Attention]]
- [[Cross-Attention]]
- [[Convolution]]

## Datasets
- [[DocVQA]]
- [[RVL-CDIP]]
- [[CORD]]

## Threads
- OCR-free document understanding
- Vision-language document modeling
- Structured extraction from document images
```

- [ ] **Step 9: Create `Maps/Efficient Transformers.md`**

Create `Maps/Efficient Transformers.md` with:

```markdown
# Efficient Transformers

## Papers
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]
- [[Swin Transformer, Hierarchical Vision Transformer using Shifted Windows]]
- [[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]
- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]

## Concepts
- [[Flash Attention]]
- [[KV-Cache]]
- [[Gradient Checkpointing]]
- [[Mixed Precision (FP16 BF16)]]
- [[Quantization]]
- [[LoRA]]

## Threads
- Reducing attention cost
- Improving throughput at high resolution or long context
- Memory-efficient training and inference
```

- [ ] **Step 10: Verify maps exist**

Run:

```bash
find Maps -maxdepth 1 -type f -name '*.md' -print | sort
```

Expected output:

```text
Maps/Attention.md
Maps/Computer Vision.md
Maps/Document AI.md
Maps/Efficient Transformers.md
Maps/Large Language Models.md
Maps/Natural Language Processing.md
Maps/Retrieval.md
Maps/Vault Map.md
```

- [ ] **Step 11: Commit maps**

Run:

```bash
git add Maps
git commit -m "docs: add vault navigation maps"
```

Expected:

- Commit succeeds.

## Task 5: Update Agent Guidance

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace the repository structure bullets in `AGENTS.md`**

Update the `Repository Structure` section in `AGENTS.md` to:

```markdown
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
```

- [ ] **Step 2: Replace the `Basics Note Template` heading in `AGENTS.md`**

Change:

```markdown
## Basics Note Template
```

to:

```markdown
## Concept Note Template
```

- [ ] **Step 3: Replace the explanatory paragraph in `AGENTS.md`**

Change the paragraph beginning with `Basics are the deep-dive versions` to:

```markdown
Concept notes are the deep-dive versions of the `> [!info]- Comments` in papers. Comments = quick 2-3 line reminder. Concepts = full explanation with formulas and pseudocode. Papers link to Concepts via `[[wikilinks]]`, creating central hub nodes in the graph view.
```

- [ ] **Step 4: Apply the same three guidance edits to `CLAUDE.md`**

Repeat Steps 1 through 3 in `CLAUDE.md` using the same replacement text.

- [ ] **Step 5: Verify no old structure references remain in guidance docs**

Run:

```bash
rg -n "^- \\*\\*Computer Vision/|^- \\*\\*Natural Language Processing/|^- \\*\\*Large Language Models/|Basics/|Basics Note Template|Basics are" AGENTS.md CLAUDE.md
```

Expected:

- No output.

- [ ] **Step 6: Commit guidance updates**

Run:

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: update vault structure guidance"
```

Expected:

- Commit succeeds.

## Task 6: Verify Links, Embeds, And Scope

**Files:**
- Read: all Markdown files.
- No intended file changes unless verification exposes a concrete broken reference created by this migration.

- [ ] **Step 1: Check for old `Basics/` path references**

Run:

```bash
rg -n "Basics/" --glob '*.md' --glob '!docs/superpowers/specs/**' --glob '!docs/superpowers/plans/**'
```

Expected:

- No output.

- [ ] **Step 2: Check duplicate note basenames**

Run:

```bash
find . -path './.git' -prune -o -type f -name '*.md' -print | sed 's#^.*/##' | sort | uniq -d
```

Expected:

- No duplicate basename that would make Obsidian wikilinks ambiguous.
- If duplicates appear only because maps share names with folders, that is acceptable because folders are not Markdown notes.

- [ ] **Step 3: Check wikilink targets by basename**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

root = Path('.')
notes = {p.stem for p in root.rglob('*.md') if '.git' not in p.parts}
links = {}
pattern = re.compile(r'!?\[\[([^\]|#]+)')
for path in root.rglob('*.md'):
    if '.git' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    for match in pattern.finditer(text):
        target = match.group(1).strip()
        if '/' in target:
            target_stem = Path(target).stem
        else:
            target_stem = target
        links.setdefault(target_stem, set()).add(str(path))

missing = sorted(target for target in links if target not in notes)
for target in missing:
    print(target)
PY
```

Expected:

- Existing missing links may include datasets or paper references that do not yet have notes, such as `ImageNet`, `MS COCO`, or `ADE20K`.
- No missing links should be caused by moving `Basics/` to `Concepts/` because wikilinks are basename-based.

- [ ] **Step 4: Check Obsidian image embeds by basename**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

root = Path('.')
files = {p.name for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
pattern = re.compile(r'!\[\[([^\]|#]+)')
missing = set()
for path in root.rglob('*.md'):
    if '.git' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    for match in pattern.finditer(text):
        target = match.group(1).strip()
        name = Path(target).name
        if name not in files:
            missing.add((str(path), target))

for path, target in sorted(missing):
    print(f"{path}: {target}")
PY
```

Expected:

- No missing embeds introduced by the migration.
- Existing missing embeds, if any, should be reviewed separately before fixing.

- [ ] **Step 5: Check protected folders were not moved**

Run:

```bash
find Blog-research interactive docs media misc -maxdepth 1 -type d -print | sort
```

Expected output includes:

```text
Blog-research
docs
interactive
media
misc
```

- [ ] **Step 6: Review staged and committed migration scope**

Run:

```bash
git status --short
git log --oneline -5
```

Expected:

- Recent commits show the paper move, concept move, map addition, and guidance update.
- Unrelated pre-existing dirty files may still be present.

## Task 7: Final Review

**Files:**
- Read: migrated structure and guidance docs.
- No intended file changes unless a verification issue needs a focused fix.

- [ ] **Step 1: Review final folder shape**

Run:

```bash
find Papers Concepts Datasets Maps -maxdepth 2 -type f -name '*.md' -print | sort
```

Expected:

- Papers are under `Papers/<discipline>/`.
- Concepts are under `Concepts/<theme>/`.
- Datasets remain under `Datasets/`.
- Maps exist under `Maps/`.

- [ ] **Step 2: Review migration diff summary**

Run:

```bash
git show --stat --oneline HEAD
git log --oneline --decorate -8
```

Expected:

- The latest commits are focused and understandable.

- [ ] **Step 3: Report outcome**

Report:

- Which folders moved.
- Which maps were created.
- Whether wikilink/embed checks found migration-caused problems.
- Any pre-existing dirty files that were intentionally left untouched.
