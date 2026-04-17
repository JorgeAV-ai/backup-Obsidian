# Quantization Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current low-signal quantization article graphics with four scene-first Manim figures rendered as stable PNGs for the article.

**Architecture:** Keep all figure logic in one Manim script with shared palette, typography, and helper utilities. Write `plan.md` first, then replace the scene classes in `script.py`, render draft stills for each scene, refine composition, and export final PNGs into `misc/` with stable article-facing names.

**Tech Stack:** Python 3, Manim Community Edition v0.20+, ffmpeg, Markdown, git

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `Blog-research/manim-quantization/plan.md` | Create | narrative arc, scene briefs, palette, target output names |
| `Blog-research/manim-quantization/script.py` | Replace | all four figure scenes plus shared theme helpers |
| `misc/quantization_memory_wall.png` | Create | final still for the intro |
| `misc/quantization_intuition.png` | Create | final still for basic intuition |
| `misc/quantization_formula.png` | Create | final still for the formula section |
| `misc/quantization_outlier_distortion.png` | Create | final still for the outlier bridge |
| `docs/superpowers/specs/2026-04-17-quantization-figures-design.md` | Reference | approved visual system and scene briefs |

---

### Task 1: Write the figure production plan

**Files:**
- Create: `Blog-research/manim-quantization/plan.md`

- [ ] **Step 1: Create `plan.md` with the approved figure set**

```markdown
# Quantization Figures Plan

## Shared visual language
- light / pastel palette
- warm background
- Menlo for all text
- opacity hierarchy: 1.0 / 0.4 / 0.15

## Target outputs
- misc/quantization_memory_wall.png
- misc/quantization_intuition.png
- misc/quantization_formula.png
- misc/quantization_outlier_distortion.png

## Scene list
1. Memory Wall
2. Quantization Intuition
3. Quantization Formula
4. Outlier Distortion
```

- [ ] **Step 2: Expand the four scenes into concrete briefs**

Use these exact headings in `plan.md`:

```markdown
### Scene 1: Memory Wall
### Scene 2: Quantization Intuition
### Scene 3: Quantization Formula
### Scene 4: Outlier Distortion
```

Under each heading, write six bullets in this exact order: misconception to correct, aha moment, composition, dominant color, must show, must avoid. Fill each bullet with the exact decisions from the approved spec, not new ideas.

- [ ] **Step 3: Verify the plan file**

Run: `rg -n "^### Scene|quantization_.*\\.png" Blog-research/manim-quantization/plan.md`
Expected: four `### Scene` headings and four output filenames.

- [ ] **Step 4: Commit the production plan**

```bash
git add -- Blog-research/manim-quantization/plan.md
git commit -m "docs: add quantization figures plan"
```

---

### Task 2: Replace `script.py` with a clean shared scene system

**Files:**
- Modify: `Blog-research/manim-quantization/script.py`

- [ ] **Step 1: Replace the top of `script.py` with shared theme constants**

```python
from manim import *
import numpy as np

MONO = "Menlo"

P = {
    "BG": "#FFFBF5",
    "TEXT": "#1A1A1A",
    "BLUE": "#4A90E2",
    "GREEN": "#67B26F",
    "AMBER": "#D89B2B",
    "RED": "#D95C5C",
    "VIOLET": "#7E6BC4",
    "PEACH": "#FFD6A5",
    "MINT": "#CAFFBF",
    "SKY": "#BDE0FE",
    "CORAL": "#FFADAD",
}

def soften(mobj, opacity):
    mobj.set_opacity(opacity)
    return mobj
```

- [ ] **Step 2: Define the four new scene classes and remove the old ones**

Use these exact class names:

```python
class Scene1MemoryWall(Scene):
    pass

class Scene2QuantizationIntuition(Scene):
    pass

class Scene3QuantizationFormula(Scene):
    pass

class Scene4OutlierDistortion(Scene):
    pass
```

No legacy `Scene1PrecisionLoss`, `Scene2MemoryWall`, or `Scene3QuantFormula` classes should remain after this step.

- [ ] **Step 3: Verify class names**

Run: `rg -n "^class Scene" Blog-research/manim-quantization/script.py`
Expected: exactly four classes with the names above.

- [ ] **Step 4: Commit the clean scene scaffold**

```bash
git add -- Blog-research/manim-quantization/script.py
git commit -m "refactor: reset quantization manim scenes"
```

---

### Task 3: Implement and draft-render the memory wall and intuition scenes

**Files:**
- Modify: `Blog-research/manim-quantization/script.py`
- Create: `misc/quantization_memory_wall.png`
- Create: `misc/quantization_intuition.png`

- [ ] **Step 1: Implement `Scene1MemoryWall` as a same-model precision comparison**

Use a composition shaped like this:

```python
labels = ["70B FP16", "70B INT8", "70B INT4"]
values = [140, 70, 35]
colors = [P["CORAL"], P["AMBER"], P["GREEN"]]
```

Represent the values with one direct scale encoding such as vertical bars or stacked memory blocks. Do not draw GPUs or decorative hardware.

- [ ] **Step 2: Draft-render Scene 1**

Run: `manim -ql --format=png -s Blog-research/manim-quantization/script.py Scene1MemoryWall`
Expected: a single PNG exists somewhere under `Blog-research/manim-quantization/media/images/` with a filename beginning `Scene1MemoryWall`

- [ ] **Step 3: Implement `Scene2QuantizationIntuition` as continuous-to-discrete mapping**

Use a composition shaped like this:

```python
left_points = VGroup(*[
    Dot(point=np.array([x, np.sin(x) * 0.35, 0]), radius=0.08, color=P["PEACH"])
    for x in np.linspace(-2.5, 2.5, 18)
])
```

Map those points visually into a smaller set of horizontal steps or bins on the right. The scene must show collapse of resolution, not just motion.

- [ ] **Step 4: Draft-render Scene 2**

Run: `manim -ql --format=png -s Blog-research/manim-quantization/script.py Scene2QuantizationIntuition`
Expected: a single PNG exists somewhere under `Blog-research/manim-quantization/media/images/` with a filename beginning `Scene2QuantizationIntuition`

- [ ] **Step 5: Copy the draft renders into stable article-facing names**

```bash
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene1MemoryWall*.png' | head -1)" \
   misc/quantization_memory_wall.png
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene2QuantizationIntuition*.png' | head -1)" \
   misc/quantization_intuition.png
```

- [ ] **Step 6: Commit the first two figures**

```bash
git add -- Blog-research/manim-quantization/script.py misc/quantization_memory_wall.png misc/quantization_intuition.png
git commit -m "feat: add quantization intro figures"
```

---

### Task 4: Implement and draft-render the color-coded formula scene

**Files:**
- Modify: `Blog-research/manim-quantization/script.py`
- Create: `misc/quantization_formula.png`

- [ ] **Step 1: Implement `Scene3QuantizationFormula` with semantic color coding**

Use a `MathTex` composition split by semantic parts:

```python
formula = MathTex(
    r"W_q", r"=", r"\operatorname{RoundClip}\left(",
    r"\frac{W}{\Delta}", r"+", r"Z", r"\right)",
    font_size=42
)

formula[0].set_color(P["VIOLET"])
formula[2].set_color(P["RED"])
formula[3][1].set_color(P["BLUE"])   # W
formula[3][3].set_color(P["GREEN"])  # Delta
formula[5].set_color(P["AMBER"])     # Z
```

Add a second line for dequantization and a compact legend underneath. Keep the same colors for the same semantic roles.

- [ ] **Step 2: Add brief labels for the semantic roles**

Use label text shaped like this:

```python
legend_items = [
    ("W", "original value", P["BLUE"]),
    ("Δ", "step size", P["GREEN"]),
    ("Z", "zero-point", P["AMBER"]),
    ("Round + Clip", "map to discrete range", P["RED"]),
    ("W_q", "quantized value", P["VIOLET"]),
]
```

- [ ] **Step 3: Draft-render Scene 3**

Run: `manim -ql --format=png -s Blog-research/manim-quantization/script.py Scene3QuantizationFormula`
Expected: a single PNG exists somewhere under `Blog-research/manim-quantization/media/images/` with a filename beginning `Scene3QuantizationFormula`

- [ ] **Step 4: Copy the render into the stable article-facing name**

```bash
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene3QuantizationFormula*.png' | head -1)" \
   misc/quantization_formula.png
```

- [ ] **Step 5: Commit the formula figure**

```bash
git add -- Blog-research/manim-quantization/script.py misc/quantization_formula.png
git commit -m "feat: add quantization formula figure"
```

---

### Task 5: Implement and draft-render the outlier distortion scene

**Files:**
- Modify: `Blog-research/manim-quantization/script.py`
- Create: `misc/quantization_outlier_distortion.png`

- [ ] **Step 1: Implement `Scene4OutlierDistortion` with a normal cluster plus one outlier**

Use a composition shaped like this:

```python
normal_x = np.linspace(-2.0, 2.0, 17)
normal_dots = VGroup(*[
    Dot(point=np.array([x, 0.2 * np.sin(2 * x), 0]), radius=0.08, color=P["BLUE"])
    for x in normal_x
])
outlier = Dot(point=np.array([4.8, 1.2, 0]), radius=0.11, color=P["RED"])
```

Then show a coarse quantization grid or bucket structure that widens to include the outlier and visibly collapses the central dots into fewer levels.

- [ ] **Step 2: Add one label only if the mechanism is not visually obvious**

If needed, use one short label like:

```python
label = Text("larger Δ", font=MONO, font_size=24, color=P["GREEN"])
```

Do not explain the scene with paragraphs of on-image text.

- [ ] **Step 3: Draft-render Scene 4**

Run: `manim -ql --format=png -s Blog-research/manim-quantization/script.py Scene4OutlierDistortion`
Expected: a single PNG exists somewhere under `Blog-research/manim-quantization/media/images/` with a filename beginning `Scene4OutlierDistortion`

- [ ] **Step 4: Copy the render into the stable article-facing name**

```bash
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene4OutlierDistortion*.png' | head -1)" \
   misc/quantization_outlier_distortion.png
```

- [ ] **Step 5: Commit the outlier figure**

```bash
git add -- Blog-research/manim-quantization/script.py misc/quantization_outlier_distortion.png
git commit -m "feat: add quantization outlier figure"
```

---

### Task 6: Final render pass and verification

**Files:**
- Modify: `Blog-research/manim-quantization/script.py`
- Modify: `misc/quantization_memory_wall.png`
- Modify: `misc/quantization_intuition.png`
- Modify: `misc/quantization_formula.png`
- Modify: `misc/quantization_outlier_distortion.png`

- [ ] **Step 1: Run a production still render for all four scenes**

```bash
manim -qh --format=png -s Blog-research/manim-quantization/script.py Scene1MemoryWall
manim -qh --format=png -s Blog-research/manim-quantization/script.py Scene2QuantizationIntuition
manim -qh --format=png -s Blog-research/manim-quantization/script.py Scene3QuantizationFormula
manim -qh --format=png -s Blog-research/manim-quantization/script.py Scene4OutlierDistortion
```

Expected: four high-resolution PNG renders exist under `Blog-research/manim-quantization/media/images/`, one per scene class.

- [ ] **Step 2: Refresh the stable article-facing copies**

```bash
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene1MemoryWall*.png' | head -1)" misc/quantization_memory_wall.png
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene2QuantizationIntuition*.png' | head -1)" misc/quantization_intuition.png
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene3QuantizationFormula*.png' | head -1)" misc/quantization_formula.png
cp "$(find Blog-research/manim-quantization/media/images -name 'Scene4OutlierDistortion*.png' | head -1)" misc/quantization_outlier_distortion.png
```

- [ ] **Step 3: Verify that the article-facing files exist**

```bash
ls -1 misc/quantization_memory_wall.png \
      misc/quantization_intuition.png \
      misc/quantization_formula.png \
      misc/quantization_outlier_distortion.png
```

Expected: four lines, one per file.

- [ ] **Step 4: Verify that no legacy scene names remain in the script**

Run: `rg -n "Scene1PrecisionLoss|Scene2MemoryWall|Scene3QuantFormula" Blog-research/manim-quantization/script.py`
Expected: no output.

- [ ] **Step 5: Commit the final figure set**

```bash
git add -- Blog-research/manim-quantization/plan.md Blog-research/manim-quantization/script.py misc/quantization_memory_wall.png misc/quantization_intuition.png misc/quantization_formula.png misc/quantization_outlier_distortion.png
git commit -m "feat: finalize quantization article figures"
```
