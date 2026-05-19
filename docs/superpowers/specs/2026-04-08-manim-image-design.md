# manim-image Skill — Design Spec

**Date**: 2026-04-08
**Status**: Approved
**Skill location**: `~/.claude/skills/manim-image/`

---

## Overview

`manim-image` is a Claude Code skill for generating high-resolution static images using Manim Community Edition. It is purpose-built for static output — not a wrapper around `manim-video`. The creative standard, scene patterns, and pipeline are distinct.

**Trigger**: user asks for illustrations, diagrams, infographics, annotated formulas, or any technical static image for notes, blog posts, slides, or papers.

---

## Use Cases

- **Formulas with colors** — annotated equations, step-by-step derivations as a single image
- **Architecture diagrams** — block diagrams, pipelines, system components with arrows
- **Comparisons / visual tables** — side-by-side model comparisons, benchmark summaries
- **Geometric concepts** — visual intuition for math operations (attention, convolution, etc.)
- **Paper infographics** — key findings of a paper summarized in one image

---

## Creative Standard

> "Una imagen se lee una vez. Cada elemento gana su lugar o no existe."

- **Composition before content** — decide the layout (centered, 2-column, hierarchical) before writing any code
- **3-level opacity hierarchy**: primary 1.0, contextual 0.4, structural 0.15
- **Reading flow**: left→right, top→bottom. The eye must know where to enter
- **No decorative elements** — if an element does not explain something, remove it
- **Breathing room** — minimum `buff=0.4` between elements, edge padding `buff=0.5`

---

## Visual Style

Two themes selectable via `--theme dark|light`. Default: `dark`.

### Dark (3B1B)

| Role | Color | Hex |
|---|---|---|
| Background | Dark gray | `#1C1C1C` |
| Primary | Blue | `#58C4DD` |
| Secondary | Green | `#83C167` |
| Accent | Yellow | `#FFFF00` |
| Warning / loss | Red-pink | `#FF6B6B` |
| Text | White | `#FFFFFF` |

### Light (Warm Pastel)

| Role | Color | Hex |
|---|---|---|
| Background | Warm white | `#FFFBF5` |
| Primary | Peach-orange | `#FFD6A5` |
| Secondary | Mint green | `#CAFFBF` |
| Accent | Sky blue | `#BDE0FE` |
| Warning / loss | Coral | `#FFADAD` |
| Text | Near-black | `#1A1A1A` |

The color constants in `script.py` are always declared at the top via a theme dict — the skill injects the right palette automatically based on `--theme`:

```python
THEME = "dark"  # injected by render.sh

PALETTES = {
    "dark":  {"BG": "#1C1C1C", "C1": "#58C4DD", "C2": "#83C167",
              "C3": "#FFFF00", "WARN": "#FF6B6B", "TEXT": "#FFFFFF"},
    "light": {"BG": "#FFFBF5", "C1": "#FFD6A5", "C2": "#CAFFBF",
              "C3": "#BDE0FE", "WARN": "#FFADAD", "TEXT": "#1A1A1A"},
}
P = PALETTES[THEME]
```

Scenes always reference `P["C1"]`, `P["BG"]`, etc. — never hardcoded hex values.

### Font
Monospace font: **Menlo**. Minimum sizes: title 36px, heading 28px, body 24px, label 18px, caption 16px.

---

## Scene Patterns

Two patterns depending on image type. Both render with `manim --format=png -s`.

### Pattern A — Pure composition (`self.add`)
For: formulas, architecture diagrams, comparison tables. Everything placed at once.

```python
from manim import *

BG = "#1C1C1C"
MONO = "Menlo"

class FormulaImage(Scene):
    def construct(self):
        self.camera.background_color = BG
        # Build all mobjects
        formula = Text("W_q = RoundClip(W/Δ + Z)", font=MONO, font_size=36, color=WHITE)
        label = Text("step size", font=MONO, font_size=18, color="#83C167")
        label.next_to(formula, DOWN, buff=0.5)
        # Add everything at once — no play(), no wait()
        self.add(formula, label)
```

### Pattern B — Frozen frame (`self.play` + render `-s`)
For: states of a process, progressively built diagrams. The last frame is captured as the image.

```python
class ArchitectureImage(Scene):
    def construct(self):
        self.camera.background_color = BG
        node1 = RoundedRectangle(corner_radius=0.2, width=2.5, height=1,
                                  fill_color="#1a1a2e", fill_opacity=1)
        label1 = Text("Encoder", font=MONO, font_size=24, color="#58C4DD")
        label1.move_to(node1)
        node2 = node1.copy().shift(RIGHT * 4)
        label2 = Text("Decoder", font=MONO, font_size=24, color="#83C167")
        label2.move_to(node2)
        arrow = Arrow(node1.get_right(), node2.get_left(), color=WHITE, buff=0.1)

        self.play(FadeIn(VGroup(node1, label1)))
        self.play(FadeIn(VGroup(node2, label2)), GrowArrow(arrow))
        # render -s captures this final state as PNG
```

**Rule**: if the image represents a "state" or the final result of a process, use Pattern B. If it's a pure diagram with no temporal logic, use Pattern A.

---

## Pipeline

```
PLAN → CODE → RENDER → EXPORT
```

1. **PLAN** — write `plan.md`: what goes in the image, layout choice, destination, palette
2. **CODE** — write `script.py`: one class per image, Pattern A or B
3. **RENDER** — `scripts/render.sh script.py ImageName --dest obsidian --out ~/path/`
4. **EXPORT** — PNG copied to destination with correct naming convention

### render.sh behavior

```bash
./render.sh script.py FormulaImage --dest obsidian --out ~/vault/misc/
# → 1600×900, dark theme, copia a misc/formula_image.png

./render.sh script.py FormulaImage --dest paper --theme light --out ~/docs/figures/
# → 2480×1754, warm pastel theme, copia a figures/formula_image.png
```

- Sets resolution from preset (see table below)
- Injects `THEME = "dark"|"light"` at top of script before rendering
- Runs `manim --format=png -s script.py FormulaImage`
- Copies output PNG to `--out` path with snake_case filename
- `--theme` default: `dark`

### Resolution presets

| `--dest` | Resolution | Aspect ratio | Use case |
|---|---|---|---|
| `obsidian` | 1600×900 | 16:9 | Vault notes `![[image.png]]` |
| `blog` | 2400×1350 | 16:9 HD | Blog post header / inline |
| `slides` | 1920×1080 | Full HD | Presentation slide |
| `paper` | 2480×1754 | A4 landscape | Academic figure |

Default if `--dest` not specified: `blog`.

---

## File Structure

```
~/.claude/skills/manim-image/
  SKILL.md                    # skill entry point + trigger description
  README.md                   # user-facing docs
  scripts/
    setup.sh                  # dependency check (same as manim-video)
    render.sh                 # render + export wrapper
  references/
    composition.md            # layouts, reading flow, rule of thirds, spacing
    image-patterns.md         # Pattern A vs B, examples per image type
    export.md                 # render commands, presets, naming convention
    color-and-style.md        # 3B1B palette for static, opacity layering, typography
    mobjects-static.md        # useful mobjects subset: VGroup layout, Arrow, Brace, Text
```

---

## SKILL.md Trigger Description

```
Use when users request: static diagrams, technical illustrations, annotated formulas,
architecture diagrams, comparison visuals, paper infographics, or any high-resolution
PNG image with mathematical or technical content. NOT for animated videos — use
manim-video for those.
```

---

## Out of Scope

- Animated GIFs — use manim-video
- Light background / paper-friendly white mode — not in v1
- SVG export — not in v1
- Interactive images — not in scope

---

## Implementation Order

1. `SKILL.md` + `README.md`
2. `scripts/setup.sh` (copy from manim-video, no changes)
3. `scripts/render.sh` (new)
4. `references/composition.md`
5. `references/image-patterns.md`
6. `references/color-and-style.md`
7. `references/mobjects-static.md`
8. `references/export.md`
