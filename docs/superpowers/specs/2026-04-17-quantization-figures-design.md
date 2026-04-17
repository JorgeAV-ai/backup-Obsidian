# Quantization Figures Design Spec

**Date**: 2026-04-17  
**Status**: Approved  
**Target**: `Blog-research/manim-quantization/`  
**Article**: `Blog-research/Quantizating LLMs, How it works and what you should be concerned about.md`

---

## Goal

Replace the current quantization article figures with a coherent family of high-quality Manim scenes designed for explanation first and rendered as static images for the article.

The figures should not behave like generic blog illustrations or decorative infographics. Each one must resolve a specific conceptual bottleneck in the article and support the linear deep research explainer structure defined in [2026-04-17-quantization-blog-design.md](/Users/jorgeav/Desktop/BeatEm/backup-Obsidian/docs/superpowers/specs/2026-04-17-quantization-blog-design.md).

---

## Direction

The approved direction is **scene-first** using the `manim-video` production standard.

That means:

- each figure is designed as a full Manim scene with a clear narrative and visual hierarchy,
- scenes are composed as if they could later become animated explainers,
- the immediate output for the article is a high-quality still render from each scene,
- the design should prioritize pedagogical clarity over decorative polish.

This is explicitly not a “video-first then extract random frames” workflow, and not an ad hoc static illustration workflow. The scene itself is the source of truth.

---

## Narrative Function

The article needs four core figures. Together they should create a visual bridge across the main narrative:

1. why quantization exists,
2. what quantization does,
3. how the formula works,
4. why naive quantization fails.

If these four figures work, the reader can move from the introduction into the methods section without getting lost.

---

## Shared Visual Language

All scenes must feel like one family.

- **Palette**: light / pastel
- **Background**: warm light background, not dark
- **Typography**: monospace only, `Menlo`
- **Hierarchy**:
  - primary elements at full opacity,
  - contextual elements around `0.4`,
  - structural elements such as grids, guides, or separators around `0.15`
- **Layout discipline**: one dominant idea per scene, generous breathing room, no clutter
- **Mood**: educational, clean, intentional, slightly editorial

Per `manim-video`, scenes should vary in dominant color and layout so the set does not feel repetitive, but they must preserve semantic consistency across the whole package.

---

## Figure Set

### Figure 1: Memory Wall

### Purpose

Correct the misconception that quantization is a minor mathematical trick. This figure should make it obvious that precision determines whether a large model is practical to load at all.

### Aha Moment

The same model moves from expensive or infeasible to manageable purely by reducing representation precision.

### Composition

- clean comparison of the same model at multiple precisions such as `FP16`, `INT8`, and `INT4`,
- memory footprint shown visually through bars, blocks, or another direct scale encoding,
- minimal labels with a clear left-to-right comparison.

### Must Show

- the relative size difference between high precision and low-bit formats,
- that the change is material, not cosmetic,
- that quantization exists because the memory budget is real.

### Must Avoid

- decorative hardware drawings,
- abstract arrows with no informational payload,
- turning the scene into a GPU infographic instead of a memory comparison.

---

### Figure 2: Quantization Intuition

### Purpose

Introduce the core idea of quantization visually before the reader has to process the formula.

### Aha Moment

Many nearby continuous values collapse into fewer discrete levels.

### Composition

- continuous values on the left,
- quantized bins, steps, or discrete levels on the right,
- visual mapping from the continuous side into the discrete side,
- enough structure to imply loss of resolution without overlabeling.

### Must Show

- what it means to move from high precision to low-bit representation,
- that compression is achieved by reducing distinguishable levels,
- that the gain in compactness comes with information loss.

### Must Avoid

- overly metaphorical imagery that hides the mathematical idea,
- excessive text,
- visuals that imply only storage savings but not loss of granularity.

---

### Figure 3: Quantization Formula

### Purpose

Turn the formula into a readable object that the article can reference directly in prose.

### Aha Moment

Each symbol in the quantization pipeline has a distinct role, and the reader can follow that role visually.

### Composition

- main quantization formula as the central object,
- semantic color coding for each major term,
- concise visual legend or annotations,
- dequantization or reconstruction shown in a second line to connect the forward and reverse steps,
- error framed as a consequence of the same color-coded pipeline.

### Approved Color Semantics

- `W`: blue, representing the original high-precision value
- `Δ` / scale: green, representing step size and resolution
- `Z` / zero-point: amber, representing offset of the quantization grid
- `Round + Clip`: red, representing the forcing of the value into the discrete integer space
- `W_q`: neutral dark or soft violet, representing the quantized result

These colors must remain stable wherever the same terms reappear inside the figure.

### Must Show

- that the formula is a sequence of meaningful transformations,
- that quantization and dequantization are linked,
- that the error comes from the discrete step size and not from arbitrary noise.

### Must Avoid

- abstract symbolic decoration that replaces explanation,
- adding too many colors,
- changing color meaning between quantization and dequantization lines.

---

### Figure 4: Outlier Distortion

### Purpose

Explain why naive quantization fails in large language models.

### Aha Moment

One extreme outlier can inflate the quantization scale so much that the useful resolution for ordinary values collapses.

### Composition

- a cluster or distribution of normal values,
- one extreme outlier that clearly sits outside the regular range,
- a quantization grid or scale that widens because of the outlier,
- a final view showing the central values collapsing into too few levels.

### Must Show

- the causal relationship between outlier magnitude and coarser quantization,
- that the failure is structural rather than random,
- why more sophisticated methods are needed after the naive approach.

### Must Avoid

- relying on text to explain the entire phenomenon,
- generic histograms with no visible relation to the quantization grid,
- making the outlier dramatic without showing its mechanical consequence.

This is the most important figure in the entire article. If it works, the transition from basic math to modern methods becomes natural.

---

## Design Principles

- one scene, one conceptual bottleneck,
- geometry before algebra whenever possible,
- minimal text inside scenes,
- no element exists unless it teaches something,
- each scene should be understandable at a glance before the reader reads the paragraph beneath it,
- each scene should still reward close reading with a second layer of precision.

---

## Article Mapping

| Figure | Article section | Role |
|---|---|---|
| Memory Wall | `1. Introduction` | justify why quantization exists |
| Quantization Intuition | `2. Basic intuition` | introduce the mapping from continuous to discrete |
| Quantization Formula | `2. Basic intuition` | explain the mechanism in symbolic form |
| Outlier Distortion | `3. Why naive quantization fails` | create the bridge into modern methods |

The comparison of modern methods should remain a table unless implementation later shows a genuine need for a fifth figure.

---

## File Map

| File | Role |
|---|---|
| `Blog-research/manim-quantization/plan.md` | scene-by-scene implementation notes |
| `Blog-research/manim-quantization/script.py` | Manim scenes for all four figures |
| `docs/superpowers/specs/2026-04-17-quantization-figures-design.md` | this design spec |
| `docs/superpowers/specs/2026-04-17-quantization-blog-design.md` | article-level redesign spec |

---

## Constraints

- one class per scene,
- every scene independently renderable,
- use monospace text only,
- preserve a shared palette and layout system across all figures,
- render high-quality stills suitable for Obsidian embedding,
- do not inherit the current low-signal figures just because assets already exist,
- do not create extra figures unless they solve a real narrative gap.

---

## Success Criteria

The figure redesign is successful if:

- each image has a clear narrative job inside the article,
- the four scenes feel like one visual system rather than four unrelated graphics,
- the quantization formula can be referenced in the text by color and function,
- the outlier figure makes naive quantization failure visually obvious,
- the visuals increase comprehension instead of merely improving aesthetics.

---

## Non-Goals

- producing a full stitched explainer video in this phase,
- illustrating every section of the article,
- rendering hardware marketing visuals,
- maximizing visual effects at the expense of clarity,
- creating diagrams whose meaning depends on long captions.

---

## Resolved Decisions

- **Workflow**: scene-first, then render static images
- **Tooling standard**: `manim-video`
- **Palette**: light / pastel
- **Figure count**: four required core figures
- **Comparison of methods**: keep as table unless a real gap remains
- **Formula strategy**: color-coded semantic explanation
- **Most critical scene**: outlier distortion
