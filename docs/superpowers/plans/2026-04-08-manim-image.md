# manim-image Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `~/.claude/skills/manim-image/` — a purpose-built skill for generating high-resolution static PNG images using Manim CE, with dark (3B1B) and light (warm pastel) themes and per-destination resolution presets.

**Architecture:** The skill has a `SKILL.md` entry point, a `render.sh` wrapper that injects theme + sets resolution before calling `manim --format=png -s`, and five reference docs covering composition, patterns, color, mobjects, and export. Scenes always use a `P` palette dict — never hardcoded hex values.

**Tech Stack:** Bash (render.sh), Manim Community Edition v0.20+, Python 3.10+

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `~/.claude/skills/manim-image/SKILL.md` | Create | Skill entry point, trigger description, pipeline overview |
| `~/.claude/skills/manim-image/README.md` | Create | User-facing docs: modes, prerequisites, usage |
| `~/.claude/skills/manim-image/scripts/setup.sh` | Copy from manim-video | Dependency check: Python, Manim, LaTeX, ffmpeg |
| `~/.claude/skills/manim-image/scripts/render.sh` | Create | --dest/--theme/--out args, resolution inject, manim call, PNG copy |
| `~/.claude/skills/manim-image/references/composition.md` | Create | Layouts, reading flow, spacing rules, 3-level opacity |
| `~/.claude/skills/manim-image/references/image-patterns.md` | Create | Pattern A vs B decision guide, full examples per image type |
| `~/.claude/skills/manim-image/references/color-and-style.md` | Create | Dark + light palettes, P dict pattern, typography scale |
| `~/.claude/skills/manim-image/references/mobjects-static.md` | Create | Useful mobjects for static images: VGroup, Arrow, Brace, Text, RoundedRectangle |
| `~/.claude/skills/manim-image/references/export.md` | Create | render.sh usage, resolution presets table, naming convention, dest-specific tips |

---

## Task 1: Scaffold directory + SKILL.md

**Files:**
- Create: `~/.claude/skills/manim-image/SKILL.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p ~/.claude/skills/manim-image/scripts
mkdir -p ~/.claude/skills/manim-image/references
```

Expected: no output, directories exist.

- [ ] **Step 2: Write SKILL.md**

```markdown
---
name: manim-image
description: "Purpose-built pipeline for generating high-resolution static PNG images using Manim Community Edition. Creates annotated formulas, architecture diagrams, comparison visuals, geometric concept illustrations, and paper infographics. Supports dark (3B1B) and light (warm pastel) themes with per-destination resolution presets (Obsidian, blog, slides, paper). Use when users request: static diagrams, technical illustrations, annotated formulas, architecture diagrams, comparison visuals, paper infographics, or any high-resolution PNG image with mathematical or technical content. NOT for animated videos — use manim-video for those."
version: 1.0.0
---

# manim-image — Static Image Production Pipeline

## Creative Standard

> "Una imagen se lee una vez. Cada elemento gana su lugar o no existe."

A static image is judged all at once — not over time. Every design decision must serve readability in a single glance.

**Composition before content.** Decide the layout (centered, 2-column, hierarchical) before writing a single line of code. A diagram built without a layout plan is always cluttered.

**3-level opacity hierarchy.** Primary elements at 1.0. Contextual elements (labels, annotations) at 0.4. Structural elements (grids, borders) at 0.15. The eye processes salience layers automatically — use this.

**Reading flow is non-negotiable.** Left→right, top→bottom. The viewer's eye must know where to enter the image and where to exit. Centered layouts are safe. Off-center layouts require intentional anchoring.

**No decorative elements.** If it doesn't explain something, it doesn't exist. No gradients, no drop shadows, no ornamental lines.

**Breathing room.** Minimum `buff=0.4` between elements. Edge padding `buff=0.5`. Crowded images look unprofessional and are harder to read.

## Themes

Two themes via `--theme dark|light` (default: `dark`):

- **dark** — 3B1B palette: `#1C1C1C` background, blue/green/yellow accent colors
- **light** — Warm pastel: `#FFFBF5` background, peach/mint/sky/coral accent colors

Scenes NEVER hardcode hex values. Always use the `P` palette dict. See `references/color-and-style.md`.

## Image Types

| Type | Pattern | Reference |
|---|---|---|
| Annotated formula | A | `references/image-patterns.md` |
| Architecture diagram | A or B | `references/image-patterns.md` |
| Comparison / table | A | `references/image-patterns.md` |
| Geometric concept | B | `references/image-patterns.md` |
| Paper infographic | A | `references/image-patterns.md` |

## Pipeline

```
PLAN → CODE → RENDER → EXPORT
```

1. **PLAN** — decide layout, theme, destination, content hierarchy
2. **CODE** — `script.py`: one class per image, Pattern A or B
3. **RENDER** — `scripts/render.sh script.py ClassName --dest obsidian --out ~/path/`
4. **EXPORT** — PNG at correct resolution in destination folder

## Prerequisites

Python 3.10+, Manim CE v0.20+ (`pip install manim`), LaTeX (optional, for MathTex), ffmpeg.

```bash
bash ~/.claude/skills/manim-image/scripts/setup.sh
```

## References

| File | Contents |
|---|---|
| `references/composition.md` | Layouts, reading flow, rule of thirds, spacing |
| `references/image-patterns.md` | Pattern A vs B, full code examples per image type |
| `references/color-and-style.md` | Dark + light palettes, P dict, typography scale |
| `references/mobjects-static.md` | Mobjects useful for static images |
| `references/export.md` | render.sh args, resolution presets, naming convention |
```

- [ ] **Step 3: Verify**

```bash
cat ~/.claude/skills/manim-image/SKILL.md | head -5
```

Expected: `---` frontmatter with `name: manim-image`.

---

## Task 2: README.md

**Files:**
- Create: `~/.claude/skills/manim-image/README.md`

- [ ] **Step 1: Write README.md**

```markdown
# manim-image Skill

Generates high-resolution static PNG images using [Manim Community Edition](https://www.manim.community/).

## What it does

Creates publication-quality static images for Obsidian notes, blog posts, slides, and papers. Not for videos — use `manim-video` for animations.

## Use cases

- **Annotated formulas** — color-coded equations with labeled components
- **Architecture diagrams** — encoder/decoder blocks, pipelines, system flows
- **Comparison visuals** — side-by-side model tables, benchmark summaries
- **Geometric concepts** — attention patterns, convolution intuition, vector spaces
- **Paper infographics** — key findings of a paper in one image

## Prerequisites

Python 3.10+, Manim CE (`pip install manim`), ffmpeg.

```bash
bash scripts/setup.sh
```

## Usage

```bash
# Dark theme (default), blog resolution (2400×1350)
scripts/render.sh script.py MyImage --out ~/output/

# Obsidian note, dark theme
scripts/render.sh script.py MyImage --dest obsidian --out ~/vault/misc/

# Paper figure, light warm-pastel theme
scripts/render.sh script.py MyImage --dest paper --theme light --out ~/docs/figures/
```

## Destination presets

| `--dest` | Resolution | Use case |
|---|---|---|
| `obsidian` | 1600×900 | Vault notes `![[image.png]]` |
| `blog` | 2400×1350 | Blog post (default) |
| `slides` | 1920×1080 | Presentation |
| `paper` | 2480×1754 | Academic figure (A4 landscape) |

## Scene template

```python
from manim import *

MONO = "Menlo"
THEME = "dark"  # injected by render.sh — do not change manually

PALETTES = {
    "dark":  {"BG": "#1C1C1C", "C1": "#58C4DD", "C2": "#83C167",
              "C3": "#FFFF00", "WARN": "#FF6B6B", "TEXT": "#FFFFFF"},
    "light": {"BG": "#FFFBF5", "C1": "#FFD6A5", "C2": "#CAFFBF",
              "C3": "#BDE0FE", "WARN": "#FFADAD", "TEXT": "#1A1A1A"},
}
P = PALETTES[THEME]

class MyImage(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        title = Text("My Diagram", font_size=36, color=P["C1"],
                     weight=BOLD, font=MONO)
        self.add(title)
```
```

- [ ] **Step 2: Verify**

```bash
ls ~/.claude/skills/manim-image/
```

Expected: `SKILL.md  README.md  references/  scripts/`

---

## Task 3: scripts/setup.sh + scripts/render.sh

**Files:**
- Copy: `~/.claude/skills/manim-image/scripts/setup.sh`
- Create: `~/.claude/skills/manim-image/scripts/render.sh`

- [ ] **Step 1: Copy setup.sh from manim-video**

```bash
cp ~/.claude/skills/manim-video/scripts/setup.sh \
   ~/.claude/skills/manim-image/scripts/setup.sh
chmod +x ~/.claude/skills/manim-image/scripts/setup.sh
```

- [ ] **Step 2: Write render.sh**

```bash
#!/usr/bin/env bash
# render.sh — render a manim-image scene to PNG
# Usage: render.sh <script.py> <ClassName> [--dest obsidian|blog|slides|paper] [--theme dark|light] [--out /path/]
set -euo pipefail

SCRIPT=""
CLASS=""
DEST="blog"
THEME="dark"
OUT_DIR="."

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)   DEST="$2";    shift 2 ;;
    --theme)  THEME="$2";   shift 2 ;;
    --out)    OUT_DIR="$2"; shift 2 ;;
    *)
      if [[ -z "$SCRIPT" ]]; then SCRIPT="$1"
      elif [[ -z "$CLASS" ]]; then CLASS="$1"
      fi
      shift ;;
  esac
done

if [[ -z "$SCRIPT" || -z "$CLASS" ]]; then
  echo "Usage: render.sh <script.py> <ClassName> [--dest obsidian|blog|slides|paper] [--theme dark|light] [--out /path/]"
  exit 1
fi

# Resolution presets
case "$DEST" in
  obsidian) WIDTH=1600;  HEIGHT=900  ;;
  blog)     WIDTH=2400;  HEIGHT=1350 ;;
  slides)   WIDTH=1920;  HEIGHT=1080 ;;
  paper)    WIDTH=2480;  HEIGHT=1754 ;;
  *)        echo "Unknown --dest '$DEST'. Use: obsidian|blog|slides|paper"; exit 1 ;;
esac

# Default theme for paper is light
if [[ "$DEST" == "paper" && "$THEME" == "dark" ]]; then
  echo "Note: --dest paper defaults to --theme light. Override with --theme dark if intended."
  THEME="light"
fi

echo "Rendering $CLASS from $SCRIPT"
echo "  dest=$DEST  theme=$THEME  resolution=${WIDTH}x${HEIGHT}  out=$OUT_DIR"

# Inject THEME into a temp copy of the script
TMPSCRIPT=$(mktemp /tmp/manim_image_XXXXXX.py)
# Replace or prepend THEME = "..."
if grep -q '^THEME = ' "$SCRIPT"; then
  sed "s|^THEME = .*|THEME = \"${THEME}\"|" "$SCRIPT" > "$TMPSCRIPT"
else
  echo "THEME = \"${THEME}\"" | cat - "$SCRIPT" > "$TMPSCRIPT"
fi

# Render
manim --format=png -s \
  --resolution "${WIDTH},${HEIGHT}" \
  --output_file "${CLASS}" \
  "$TMPSCRIPT" "$CLASS"

rm -f "$TMPSCRIPT"

# Find the rendered PNG (Manim puts it in media/images/<script_stem>/)
SCRIPT_STEM=$(basename "$TMPSCRIPT" .py)
PNG_PATH=$(find media/images -name "${CLASS}.png" 2>/dev/null | head -1)

if [[ -z "$PNG_PATH" ]]; then
  echo "Error: could not find rendered PNG. Check Manim output above."
  exit 1
fi

# Snake-case output filename
OUT_NAME=$(echo "$CLASS" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]').png
mkdir -p "$OUT_DIR"
cp "$PNG_PATH" "${OUT_DIR}/${OUT_NAME}"
echo "Saved: ${OUT_DIR}/${OUT_NAME}"
```

- [ ] **Step 3: Make executable**

```bash
chmod +x ~/.claude/skills/manim-image/scripts/render.sh
```

- [ ] **Step 4: Smoke test render.sh help**

```bash
bash ~/.claude/skills/manim-image/scripts/render.sh 2>&1 || true
```

Expected output contains: `Usage: render.sh <script.py> <ClassName>`

---

## Task 4: references/composition.md

**Files:**
- Create: `~/.claude/skills/manim-image/references/composition.md`

- [ ] **Step 1: Write composition.md**

```markdown
# Composition Reference

## Core Principle

Design the layout before writing any code. A diagram built without a composition plan is always cluttered.

Answer these three questions first:
1. **What is the single most important element?** It gets the center or top-left anchor.
2. **What is the reading order?** Left→right, top→bottom by default.
3. **What can be removed?** Every element that doesn't explain something.

---

## Layout Templates

### Centered (single focal point)
Best for: formulas, single concept, title card.

```
┌─────────────────────────────┐
│                             │
│        [MAIN ELEMENT]       │
│                             │
│    [label]       [label]    │
│                             │
└─────────────────────────────┘
```

```python
main = Text("W_q = RoundClip(W/Δ + Z)", font_size=36, color=P["C1"], font=MONO)
main.move_to(ORIGIN)
label_left  = Text("weight", font_size=18, color=P["C2"], font=MONO)
label_right = Text("step size", font_size=18, color=P["C3"], font=MONO)
label_left.next_to(main,  DL, buff=0.5)
label_right.next_to(main, DR, buff=0.5)
self.add(main, label_left, label_right)
```

### Two-column (comparison or before/after)
Best for: comparisons, side-by-side architectures.

```
┌─────────────────────────────┐
│  [TITLE]                    │
│  ┌──────────┐ ┌──────────┐  │
│  │  LEFT    │ │  RIGHT   │  │
│  │  PANEL   │ │  PANEL   │  │
│  └──────────┘ └──────────┘  │
│  [caption]     [caption]    │
└─────────────────────────────┘
```

```python
left_panel  = VGroup(...)  # build left content
right_panel = VGroup(...)  # build right content
cols = VGroup(left_panel, right_panel).arrange(RIGHT, buff=1.5)
cols.move_to(ORIGIN).shift(DOWN * 0.3)
title = Text("Title", font_size=36, color=P["TEXT"], font=MONO, weight=BOLD)
title.next_to(cols, UP, buff=0.5)
self.add(title, cols)
```

### Hierarchical (top-down flow)
Best for: pipelines, step-by-step processes, architecture flows.

```
┌─────────────────────────────┐
│        [STEP 1]             │
│           ↓                 │
│        [STEP 2]             │
│           ↓                 │
│        [STEP 3]             │
└─────────────────────────────┘
```

```python
steps = VGroup(block1, block2, block3).arrange(DOWN, buff=0.6)
arrows = VGroup(
    Arrow(block1.get_bottom(), block2.get_top(), buff=0.05, color=P["TEXT"]),
    Arrow(block2.get_bottom(), block3.get_top(), buff=0.05, color=P["TEXT"]),
)
self.add(steps, arrows)
```

---

## Opacity Layering

Never show everything at full brightness. The brain processes salience layers automatically.

| Layer | Opacity | Use for |
|---|---|---|
| Primary | 1.0 | The one thing the image is about |
| Contextual | 0.4 | Labels, annotations, supporting info |
| Structural | 0.15 | Grid lines, borders, background shapes |

```python
# Primary
main_formula.set_opacity(1.0)
# Contextual label
annotation.set_opacity(0.4)
# Structural background
background_rect.set_opacity(0.15)
```

---

## Spacing Rules

- `buff=0.4` minimum between any two elements
- `buff=0.5` for edge padding (`.to_edge(DOWN, buff=0.5)`)
- `buff=0.6–0.8` between major sections (title vs content)
- Never let text touch the frame edge

## Rule of Thirds

For non-centered layouts, place the primary element at one of the four intersection points:

```
─────┬─────┬─────
     │     │
─────┼─────┼─────   ← place key element at a ┼ intersection
     │     │
─────┴─────┴─────
```

```python
# Upper-left third
element.move_to(np.array([-config.frame_width/3,
                            config.frame_height/3, 0]))
```
```

- [ ] **Step 2: Verify**

```bash
wc -l ~/.claude/skills/manim-image/references/composition.md
```

Expected: > 80 lines.

---

## Task 5: references/image-patterns.md

**Files:**
- Create: `~/.claude/skills/manim-image/references/image-patterns.md`

- [ ] **Step 1: Write image-patterns.md**

```markdown
# Image Patterns Reference

## Pattern A vs Pattern B

| | Pattern A | Pattern B |
|---|---|---|
| Code | `self.add()` only | `self.play()` + render `-s` |
| Use when | Pure diagram, no temporal logic | Image represents a state or built-up process |
| Speed | Faster (no animation rendering) | Slower (renders full animation, saves last frame) |
| Examples | Formula, comparison table, static architecture | Graph after construction, transformer after attention |

**Decision rule**: does the image show something that was *built* or *transformed*? → Pattern B. Is it a static composition of labeled elements? → Pattern A.

---

## Pattern A Examples

### Annotated Formula

```python
from manim import *
MONO = "Menlo"
THEME = "dark"
PALETTES = {
    "dark":  {"BG": "#1C1C1C", "C1": "#58C4DD", "C2": "#83C167",
              "C3": "#FFFF00", "WARN": "#FF6B6B", "TEXT": "#FFFFFF"},
    "light": {"BG": "#FFFBF5", "C1": "#FFD6A5", "C2": "#CAFFBF",
              "C3": "#BDE0FE", "WARN": "#FFADAD", "TEXT": "#1A1A1A"},
}
P = PALETTES[THEME]

class QuantFormula(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]

        # Formula parts with per-component colors
        wq    = Text("W_q",         font_size=40, color=P["TEXT"],  font=MONO, weight=BOLD)
        eq    = Text(" = ",          font_size=40, color=P["TEXT"],  font=MONO)
        clip  = Text("RoundClip(",  font_size=40, color=P["WARN"],  font=MONO, weight=BOLD)
        w     = Text("W",            font_size=40, color=P["C1"],    font=MONO, weight=BOLD)
        div   = Text("/",            font_size=40, color=P["TEXT"],  font=MONO)
        delta = Text("Δ",            font_size=40, color=P["C2"],    font=MONO, weight=BOLD)
        plus  = Text(" + ",          font_size=40, color=P["TEXT"],  font=MONO)
        z     = Text("Z",            font_size=40, color=P["C3"],    font=MONO, weight=BOLD)
        close = Text(", MIN, MAX)",  font_size=40, color=P["WARN"],  font=MONO, weight=BOLD)

        formula = VGroup(wq, eq, clip, w, div, delta, plus, z, close).arrange(RIGHT, buff=0.05)
        formula.move_to(ORIGIN).shift(UP * 0.5)

        # Labels with arrows
        lbl_w = Text("original weight", font_size=18, color=P["C1"], font=MONO)
        lbl_w.next_to(w, DOWN, buff=0.8)
        arr_w = Arrow(lbl_w.get_top(), w.get_bottom(), color=P["C1"],
                      stroke_width=2, buff=0.05, max_tip_length_to_length_ratio=0.2)

        lbl_d = Text("step size", font_size=18, color=P["C2"], font=MONO)
        lbl_d.next_to(delta, DOWN, buff=0.8)
        arr_d = Arrow(lbl_d.get_top(), delta.get_bottom(), color=P["C2"],
                      stroke_width=2, buff=0.05, max_tip_length_to_length_ratio=0.2)

        lbl_z = Text("zero point", font_size=18, color=P["C3"], font=MONO)
        lbl_z.next_to(z, DOWN, buff=0.8)
        arr_z = Arrow(lbl_z.get_top(), z.get_bottom(), color=P["C3"],
                      stroke_width=2, buff=0.05, max_tip_length_to_length_ratio=0.2)

        self.add(formula, lbl_w, arr_w, lbl_d, arr_d, lbl_z, arr_z)
```

### Comparison Table

```python
class ModelComparison(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]

        headers = ["Method", "Recall@10", "QPS", "RAM"]
        rows = [
            ["HNSW",    "99.1%", "6,700",  "512 GB"],
            ["DiskANN", "95.4%", "5,200",  "64 GB"],
            ["IVF-PQ",  "87.3%", "12,000", "64 GB"],
        ]
        col_w = [2.2, 1.6, 1.6, 1.4]
        row_h = 0.6

        table_group = VGroup()
        for r, row in enumerate([headers] + rows):
            for c, cell in enumerate(row):
                color = P["C1"] if r == 0 else P["TEXT"]
                weight = BOLD if r == 0 else NORMAL
                opacity = 1.0 if r == 0 else (1.0 if r % 2 == 1 else 0.7)
                t = Text(cell, font_size=20, color=color, font=MONO, weight=weight)
                t.set_opacity(opacity)
                x = sum(col_w[:c]) - sum(col_w) / 2 + col_w[c] / 2
                y = -r * row_h
                t.move_to(np.array([x, y, 0]))
                table_group.add(t)

        # Horizontal divider under header
        divider = Line(
            LEFT * sum(col_w) / 2, RIGHT * sum(col_w) / 2,
            color=P["C1"], stroke_width=1.5
        ).next_to(table_group[len(headers) - 1], DOWN, buff=0.15)

        table_group.move_to(ORIGIN)
        self.add(table_group, divider)
```

---

## Pattern B Examples

### Architecture Diagram (built up)

```python
class EncoderDecoderDiagram(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        MONO = "Menlo"

        def make_block(label, color, w=2.5, h=1.0):
            rect = RoundedRectangle(corner_radius=0.15, width=w, height=h,
                                    fill_color=P["BG"], fill_opacity=1,
                                    stroke_color=color, stroke_width=2)
            txt = Text(label, font_size=22, color=color, font=MONO, weight=BOLD)
            txt.move_to(rect)
            return VGroup(rect, txt)

        encoder = make_block("Encoder", P["C1"])
        decoder = make_block("Decoder", P["C2"]).shift(RIGHT * 4)
        output  = make_block("Output",  P["C3"]).shift(RIGHT * 8)

        arrow1 = Arrow(encoder.get_right(), decoder.get_left(),
                       color=P["TEXT"], buff=0.1, stroke_width=2)
        arrow2 = Arrow(decoder.get_right(), output.get_left(),
                       color=P["TEXT"], buff=0.1, stroke_width=2)

        lbl1 = Text("hidden state", font_size=16, color=P["TEXT"], font=MONO)
        lbl1.set_opacity(0.5).next_to(arrow1, UP, buff=0.1)
        lbl2 = Text("logits", font_size=16, color=P["TEXT"], font=MONO)
        lbl2.set_opacity(0.5).next_to(arrow2, UP, buff=0.1)

        diagram = VGroup(encoder, decoder, output, arrow1, arrow2, lbl1, lbl2)
        diagram.move_to(ORIGIN)

        # Pattern B: play to build up, -s captures last frame
        self.play(FadeIn(encoder))
        self.play(GrowArrow(arrow1), FadeIn(lbl1))
        self.play(FadeIn(decoder))
        self.play(GrowArrow(arrow2), FadeIn(lbl2))
        self.play(FadeIn(output))
```

---

## Image Type Quick Reference

| Type | Pattern | Layout | Key mobjects |
|---|---|---|---|
| Annotated formula | A | Centered | Text parts + Arrow + Text labels |
| Architecture diagram | A or B | Hierarchical / horizontal | RoundedRectangle, Arrow, VGroup |
| Comparison table | A | Grid | Text, Line (dividers), VGroup |
| Geometric concept | B | Centered | Axes, Polygon, Arrow, TracedPath |
| Paper infographic | A | Two-column | VGroup, SurroundingRectangle, Text |
```

- [ ] **Step 2: Verify**

```bash
wc -l ~/.claude/skills/manim-image/references/image-patterns.md
```

Expected: > 120 lines.

---

## Task 6: references/color-and-style.md

**Files:**
- Create: `~/.claude/skills/manim-image/references/color-and-style.md`

- [ ] **Step 1: Write color-and-style.md**

```markdown
# Color and Style Reference

## The P Dict Pattern

Scenes NEVER hardcode hex values. Always declare the palette dict at the top of the script and reference `P["KEY"]` throughout:

```python
THEME = "dark"  # injected by render.sh — do not change manually

PALETTES = {
    "dark":  {"BG": "#1C1C1C", "C1": "#58C4DD", "C2": "#83C167",
              "C3": "#FFFF00", "WARN": "#FF6B6B", "TEXT": "#FFFFFF"},
    "light": {"BG": "#FFFBF5", "C1": "#FFD6A5", "C2": "#CAFFBF",
              "C3": "#BDE0FE", "WARN": "#FFADAD", "TEXT": "#1A1A1A"},
}
P = PALETTES[THEME]
MONO = "Menlo"
```

This makes the scene theme-agnostic. `render.sh` injects `THEME = "dark"` or `THEME = "light"` before running Manim.

---

## Dark Theme (3B1B)

Background: `#1C1C1C`

| Key | Role | Hex | Use for |
|---|---|---|---|
| `P["C1"]` | Primary blue | `#58C4DD` | Main formula, key concept, titles |
| `P["C2"]` | Secondary green | `#83C167` | Supporting elements, secondary labels |
| `P["C3"]` | Accent yellow | `#FFFF00` | Highlights, zero point, emphasis |
| `P["WARN"]` | Warning red-pink | `#FF6B6B` | Errors, losses, clipping operations |
| `P["TEXT"]` | White | `#FFFFFF` | Body text, operators, neutral elements |

---

## Light Theme (Warm Pastel)

Background: `#FFFBF5`

| Key | Role | Hex | Use for |
|---|---|---|---|
| `P["C1"]` | Peach-orange | `#FFD6A5` | Main formula, key concept, titles |
| `P["C2"]` | Mint green | `#CAFFBF` | Supporting elements, secondary labels |
| `P["C3"]` | Sky blue | `#BDE0FE` | Highlights, emphasis |
| `P["WARN"]` | Coral | `#FFADAD` | Errors, losses, warnings |
| `P["TEXT"]` | Near-black | `#1A1A1A` | Body text, operators, neutral elements |

> **Note on light theme contrast**: pastel fills are for backgrounds/highlights. For readable text on light background, use `P["TEXT"]` (`#1A1A1A`) or darken the pastel by ~40%:
> ```python
> # Readable label on light bg
> Text("label", color=P["TEXT"], font_size=18)
> # Colored fill box
> Rectangle(fill_color=P["C1"], fill_opacity=0.4)
> ```

---

## Opacity Layering

| Level | Opacity | Use for |
|---|---|---|
| Primary | `1.0` | The one thing the image explains |
| Contextual | `0.4` | Labels, annotations, supporting info |
| Structural | `0.15` | Borders, background shapes, grid lines |

```python
main_element.set_opacity(1.0)
annotation.set_opacity(0.4)
background_rect.set_opacity(0.15)
```

---

## Typography Scale

| Role | `font_size` | `weight` | Use for |
|---|---|---|---|
| Title | 40–48 | `BOLD` | Image title, main heading |
| Heading | 28–36 | `BOLD` | Section labels |
| Body | 22–28 | `NORMAL` | Explanatory text, formula parts |
| Label | 16–20 | `NORMAL` | Annotations, axis labels |
| Caption | 14–16 | `NORMAL` | Fine print, source credits |

**Minimum**: never below `font_size=14`. Below that, unreadable at any resolution.

**Font**: always `font=MONO` (Menlo). Manim's Pango renderer produces broken kerning with proportional fonts.

---

## Common Mistakes

```python
# WRONG — hardcoded hex, breaks theme switching
title = Text("Hello", color="#58C4DD")

# RIGHT — theme-agnostic
title = Text("Hello", color=P["C1"])

# WRONG — text too small
label = Text("annotation", font_size=10)

# RIGHT
label = Text("annotation", font_size=16)

# WRONG — no opacity layering, everything screams at the viewer
bg_rect.set_opacity(1.0)
annotation.set_opacity(1.0)

# RIGHT
bg_rect.set_opacity(0.15)
annotation.set_opacity(0.4)
```
```

- [ ] **Step 2: Verify**

```bash
grep -c "P\[" ~/.claude/skills/manim-image/references/color-and-style.md
```

Expected: >= 10 occurrences of `P["..."]` pattern.

---

## Task 7: references/mobjects-static.md

**Files:**
- Create: `~/.claude/skills/manim-image/references/mobjects-static.md`

- [ ] **Step 1: Write mobjects-static.md**

```markdown
# Mobjects for Static Images

Focused subset of Manim mobjects most useful for static image composition. For the full reference, see manim-video's `references/mobjects.md`.

---

## Text

```python
# Body text
Text("label", font_size=24, color=P["TEXT"], font=MONO)

# Bold title
Text("Title", font_size=40, color=P["C1"], font=MONO, weight=BOLD)

# Multi-line (use \n)
Text("line 1\nline 2", font_size=20, color=P["TEXT"], font=MONO)
```

**Always** use `font=MONO` (Menlo). Proportional fonts produce broken kerning.

---

## VGroup — layout workhorse

```python
# Horizontal row
row = VGroup(elem1, elem2, elem3).arrange(RIGHT, buff=0.4)

# Vertical stack
col = VGroup(title, body, caption).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

# Grid: stack rows
grid = VGroup(row1, row2, row3).arrange(DOWN, buff=0.5)

# Center on screen
group.move_to(ORIGIN)

# Align to edge with padding
group.to_edge(UP, buff=0.5)
group.to_edge(DOWN, buff=0.5)
```

---

## Arrow

```python
# Simple arrow between two mobjects
arrow = Arrow(source.get_right(), target.get_left(),
              color=P["TEXT"], buff=0.1, stroke_width=2,
              max_tip_length_to_length_ratio=0.15)

# Short annotation arrow (label → element)
arr = Arrow(label.get_top(), element.get_bottom(),
            color=P["C1"], stroke_width=1.5, buff=0.05,
            max_tip_length_to_length_ratio=0.2)
```

---

## RoundedRectangle — for blocks/nodes

```python
block = RoundedRectangle(
    corner_radius=0.15,
    width=2.5, height=1.0,
    fill_color=P["BG"],      # same as background = "hollow"
    fill_opacity=1,
    stroke_color=P["C1"],
    stroke_width=2
)
label = Text("Encoder", font_size=22, color=P["C1"], font=MONO, weight=BOLD)
label.move_to(block)
node = VGroup(block, label)
```

---

## SurroundingRectangle — highlight/callout

```python
box = SurroundingRectangle(
    target_mobject,
    color=P["WARN"],
    buff=0.15,
    stroke_width=1.5,
    corner_radius=0.08
)
self.add(box)
```

---

## Brace — span label

```python
brace = Brace(formula_part, DOWN, color=P["C2"])
brace_label = Text("step size", font_size=16, color=P["C2"], font=MONO)
brace_label.next_to(brace, DOWN, buff=0.15)
self.add(brace, brace_label)
```

**Note**: do NOT use `brace.get_text()` — it calls LaTeX. Use `Text()` + `next_to()` instead.

---

## Line — dividers and separators

```python
# Horizontal divider
divider = Line(LEFT * 3, RIGHT * 3, color=P["TEXT"], stroke_width=1)
divider.set_opacity(0.3)

# Fraction bar
frac_line = Line(LEFT * 0.4, RIGHT * 0.4, color=P["C1"], stroke_width=2)
```

---

## Positioning cheatsheet

```python
elem.move_to(ORIGIN)              # center of screen
elem.to_edge(UP, buff=0.5)        # top edge
elem.to_edge(DOWN, buff=0.5)      # bottom edge
elem.to_edge(LEFT, buff=0.5)      # left edge
elem.next_to(other, RIGHT, buff=0.4)   # right of other
elem.next_to(other, DOWN,  buff=0.3)   # below other
elem.next_to(other, UP,    buff=0.3)   # above other
elem.align_to(other, LEFT)        # align left edges
elem.shift(RIGHT * 2 + UP * 0.5)  # relative shift
```
```

- [ ] **Step 2: Verify**

```bash
grep -c "def \|class \|```" ~/.claude/skills/manim-image/references/mobjects-static.md
```

Expected: > 10.

---

## Task 8: references/export.md

**Files:**
- Create: `~/.claude/skills/manim-image/references/export.md`

- [ ] **Step 1: Write export.md**

```markdown
# Export Reference

## render.sh — the main tool

```bash
# Full usage
scripts/render.sh <script.py> <ClassName> \
  [--dest obsidian|blog|slides|paper] \
  [--theme dark|light] \
  [--out /output/directory/]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `<script.py>` | required | Path to the Python script |
| `<ClassName>` | required | Scene class name to render |
| `--dest` | `blog` | Destination preset (sets resolution) |
| `--theme` | `dark` | Color theme (`dark` or `light`) |
| `--out` | `.` (current dir) | Output directory for the PNG |

> **Note**: `--dest paper` automatically switches to `--theme light` unless `--theme dark` is explicitly passed.

---

## Resolution Presets

| `--dest` | Width | Height | Aspect | Use case |
|---|---|---|---|---|
| `obsidian` | 1600 | 900 | 16:9 | Vault notes: `![[image.png]]` |
| `blog` | 2400 | 1350 | 16:9 HD | Blog post inline / header |
| `slides` | 1920 | 1080 | 16:9 FHD | Keynote / PowerPoint slide |
| `paper` | 2480 | 1754 | A4 landscape | Academic figure (300 DPI at A4) |

---

## Output Naming Convention

The output PNG is named by converting the class name from CamelCase to snake_case:

| Class name | Output file |
|---|---|
| `QuantFormula` | `quant_formula.png` |
| `EncoderDecoderDiagram` | `encoder_decoder_diagram.png` |
| `AttentionHeatmap` | `attention_heatmap.png` |

---

## Usage Examples

```bash
# Obsidian note — dark theme, 1600×900
scripts/render.sh script.py QuantFormula \
  --dest obsidian \
  --out ~/Desktop/BeatEm/backup-Obsidian/misc/
# Output: ~/Desktop/BeatEm/backup-Obsidian/misc/quant_formula.png
# Embed in note: ![[quant_formula.png]]

# Blog post — dark theme, 2400×1350 (default)
scripts/render.sh script.py AttentionDiagram --out ~/blog/images/

# Slides — dark theme, 1920×1080
scripts/render.sh script.py ModelComparison \
  --dest slides \
  --out ~/presentations/figures/

# Paper figure — light warm pastel, 2480×1754
scripts/render.sh script.py TransformerArchitecture \
  --dest paper \
  --theme light \
  --out ~/papers/arxiv-2026/figures/
```

---

## Manual Render (without render.sh)

```bash
# Draft quality PNG (fast)
manim --format=png -s --resolution 1600,900 script.py MyImage

# Production (same command — PNG quality doesn't change with -ql/-qh)
manim --format=png -s --resolution 2400,1350 script.py MyImage

# Output lands in:
# media/images/<script_stem>/MyImage.png
```

---

## Destination-Specific Tips

### Obsidian
- Embed with `![[filename.png]]` — file must be in `misc/` of the vault
- 1600×900 is sharp enough for retina displays at standard note width
- Dark theme matches Obsidian's default dark mode

### Blog
- 2400×1350 renders crisp on retina at 1200px wide display
- Use descriptive alt text when embedding

### Slides
- 1920×1080 fills a Full HD slide with no upscaling
- Keep text `font_size >= 28` — projected text reads smaller than on screen

### Paper
- 2480×1754 = A4 landscape at ~150 DPI (sufficient for digital PDF)
- For print at 300 DPI, render at 4960×3508 manually (double the preset values)
- Light theme is mandatory for readability in print
```

- [ ] **Step 2: Verify all references exist**

```bash
ls ~/.claude/skills/manim-image/references/
```

Expected: `color-and-style.md  composition.md  export.md  image-patterns.md  mobjects-static.md`

- [ ] **Step 3: Final smoke test**

Create a minimal test script and render it:

```bash
cd /tmp && cat > test_manim_image.py << 'EOF'
from manim import *
MONO = "Menlo"
THEME = "dark"
PALETTES = {
    "dark":  {"BG": "#1C1C1C", "C1": "#58C4DD", "C2": "#83C167",
              "C3": "#FFFF00", "WARN": "#FF6B6B", "TEXT": "#FFFFFF"},
    "light": {"BG": "#FFFBF5", "C1": "#FFD6A5", "C2": "#CAFFBF",
              "C3": "#BDE0FE", "WARN": "#FFADAD", "TEXT": "#1A1A1A"},
}
P = PALETTES[THEME]

class TestImage(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        title = Text("manim-image works", font_size=40,
                     color=P["C1"], font=MONO, weight=BOLD)
        sub   = Text("dark theme", font_size=24,
                     color=P["C2"], font=MONO)
        sub.next_to(title, DOWN, buff=0.4)
        self.add(title, sub)
EOF
bash ~/.claude/skills/manim-image/scripts/render.sh \
  test_manim_image.py TestImage --dest blog --out /tmp/manim_out/
```

Expected: `Saved: /tmp/manim_out/test_image.png`

- [ ] **Step 4: Commit**

```bash
cd ~/.claude
git add skills/manim-image/
git commit -m "feat: add manim-image skill — static PNG pipeline with dark/light themes"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: SKILL.md ✓, README ✓, setup.sh ✓, render.sh ✓, all 5 references ✓, dark+light palettes ✓, P dict pattern ✓, resolution presets ✓, Pattern A + B ✓
- [x] **Placeholders**: none found
- [x] **Type consistency**: `P["BG"]`, `P["C1"]`, `P["C2"]`, `P["C3"]`, `P["WARN"]`, `P["TEXT"]` used consistently across all tasks
- [x] **render.sh note**: `--dest paper` auto-switches to light theme documented in both render.sh and export.md
- [x] **Brace.get_text() warning**: documented in mobjects-static.md to prevent LaTeX errors
