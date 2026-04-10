# BatchNorm Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a short Manim explainer video for `Basics/BatchNorm.md` that uses a hybrid 3D+2D visual language to explain BatchNorm, `gamma`/`beta`, and the training vs inference statistic split.

**Architecture:** Create a new self-contained Manim project under `Blog-research/manim-batchnorm/` with one scene class per concept. Scenes 1, 2, and 4 use 3D camera motion to show activation geometry; scenes 3 and 5 use sparse 2D overlays to pin down the formulas and the closing takeaway. Keep text minimal, monospace, and out of the way of the geometry.

**Tech Stack:** Python 3.10+, Manim Community Edition, LaTeX for equations, ffmpeg for stitch/export.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `Blog-research/manim-batchnorm/script.py` | Create | All Manim scenes and shared visual constants |
| `Blog-research/manim-batchnorm/plan.md` | Create | Local scene-by-scene production notes for the project itself |
| `Blog-research/manim-batchnorm/media/` | Generated | Manim output frames and video assets |
| `Blog-research/manim-batchnorm/final.mp4` | Generated | Final stitched output, if scenes are concatenated |

---

## Task 1: Scaffold the BatchNorm video project

**Files:**
- Create: `Blog-research/manim-batchnorm/script.py`
- Create: `Blog-research/manim-batchnorm/plan.md`

- [ ] **Step 1: Create the project directory**

```bash
mkdir -p Blog-research/manim-batchnorm
```

Expected: directory exists and is empty aside from generated `media/` later.

- [ ] **Step 2: Write the shared Manim setup**

```python
from manim import *
import numpy as np

THEME = "light"

PALETTES = {
    "light": {
        "BG": "#FFFBF5",
        "TEXT": "#1A1A1A",
        "PRIMARY": "#BDE0FE",
        "SECONDARY": "#CAFFBF",
        "ACCENT": "#FFD6A5",
        "WARN": "#FFADAD",
    },
    "dark": {
        "BG": "#1C1C1C",
        "TEXT": "#FFFFFF",
        "PRIMARY": "#58C4DD",
        "SECONDARY": "#83C167",
        "ACCENT": "#FFFF00",
        "WARN": "#FF6B6B",
    },
}

P = PALETTES[THEME]
MONO = "Menlo"
```

Expected: `script.py` has one palette dict and a single font constant shared by every scene.

- [ ] **Step 3: Write the local project plan**

```markdown
# BatchNorm Manim Project

Scene order:
1. Activation cloud before normalization
2. BatchNorm recenters and rescales the batch
3. gamma/beta recover affine flexibility
4. Training statistics vs running statistics
5. Closing takeaway
```

Expected: a short local note that mirrors the implementation order and keeps the project easy to reopen later.

- [ ] **Step 4: Verify the scaffold**

```bash
python -m py_compile Blog-research/manim-batchnorm/script.py
```

Expected: no syntax errors.

---

## Task 2: Build scenes 1 and 2

**Files:**
- Modify: `Blog-research/manim-batchnorm/script.py`

- [ ] **Step 1: Write the failing render target by adding two scene classes**

```python
class Scene1_ActivationCloud(ThreeDScene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.wait(1)

class Scene2_BatchNormalize(ThreeDScene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.wait(1)
```

Expected: Manim can discover both scene names.

- [ ] **Step 2: Implement the unstable activation cloud**

```python
class Scene1_ActivationCloud(ThreeDScene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)
        axes = ThreeDAxes(x_length=5, y_length=3, z_length=3)
        points = np.array([
            [-1.9, -0.2, 0.7],
            [-1.4,  0.1, 0.5],
            [-1.1,  0.4, 0.9],
            [-0.8, -0.1, 0.6],
            [-0.5,  0.2, 0.8],
            [-0.2,  0.5, 1.0],
        ])
        cloud = VGroup(*[
            Dot3D(point=p, radius=0.05, color=P["PRIMARY"])
            for p in points
        ])
        self.add(axes, cloud)
        self.wait(2)
```

Expected: a clearly off-center, elongated 3D batch of activations.

- [ ] **Step 3: Implement the normalization morph**

```python
class Scene2_BatchNormalize(ThreeDScene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)
        before_points = np.array([
            [-1.9, -0.2, 0.7],
            [-1.4,  0.1, 0.5],
            [-1.1,  0.4, 0.9],
            [-0.8, -0.1, 0.6],
            [-0.5,  0.2, 0.8],
            [-0.2,  0.5, 1.0],
        ])
        after_points = np.array([
            [-0.9, -0.4, 0.1],
            [-0.5, -0.1, -0.1],
            [-0.2,  0.1, 0.3],
            [ 0.1, -0.2, 0.0],
            [ 0.4,  0.0, 0.2],
            [ 0.7,  0.2, 0.4],
        ])
        before = VGroup(*[
            Dot3D(point=p, radius=0.05, color=P["PRIMARY"])
            for p in before_points
        ])
        after = VGroup(*[
            Dot3D(point=p, radius=0.05, color=P["SECONDARY"])
            for p in after_points
        ])
        self.add(before)
        self.play(ReplacementTransform(before, after), run_time=2.5)
        self.wait(2)
```

Expected: the same cloud recenters and becomes more balanced in spread, with no extra text clutter.

- [ ] **Step 4: Render draft previews**

```bash
manim -ql Blog-research/manim-batchnorm/script.py Scene1_ActivationCloud Scene2_BatchNormalize
```

Expected: draft renders complete and show the geometric transformation clearly.

---

## Task 3: Build scenes 3 and 4

**Files:**
- Modify: `Blog-research/manim-batchnorm/script.py`

- [ ] **Step 1: Add the affine recovery scene**

```python
class Scene3_GammaBeta(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        formula = MathTex(r"y = \gamma \hat{x} + \beta", font_size=42)
        gamma = Text("gamma stretches", font=MONO, font_size=22, color=P["ACCENT"])
        beta = Text("beta shifts", font=MONO, font_size=22, color=P["WARN"])
        labels = VGroup(gamma, beta).arrange(DOWN, buff=0.45).next_to(formula, DOWN, buff=0.7)
        self.add(formula, labels)
        self.wait(2)
```

Expected: `gamma` and `beta` read as learnable recovery knobs, not as an extra lecture.

- [ ] **Step 2: Add the train vs inference scene**

```python
class Scene4_TrainVsInference(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        left_title = Text("training", font=MONO, font_size=24, color=P["PRIMARY"])
        right_title = Text("inference", font=MONO, font_size=24, color=P["SECONDARY"])
        left_stats = Text("batch mean / batch var", font=MONO, font_size=20, color=P["TEXT"])
        right_stats = Text("running mean / running var", font=MONO, font_size=20, color=P["TEXT"])
        left = VGroup(left_title, left_stats).arrange(DOWN, buff=0.3).shift(LEFT * 3)
        right = VGroup(right_title, right_stats).arrange(DOWN, buff=0.3).shift(RIGHT * 3)
        arrow = Arrow(left.get_right(), right.get_left(), buff=0.2, color=P["ACCENT"])
        self.add(left, right, arrow)
        self.wait(2)
```

Expected: batch statistics and running statistics are visually distinct, and the viewer can tell which one is used in each mode.

- [ ] **Step 3: Render draft previews**

```bash
manim -ql Blog-research/manim-batchnorm/script.py Scene3_GammaBeta Scene4_TrainVsInference
```

Expected: the math is readable, and the training/inference split is unambiguous without narration.

---

## Task 4: Build scene 5 and close the video

**Files:**
- Modify: `Blog-research/manim-batchnorm/script.py`

- [ ] **Step 1: Add the closing takeaway scene**

```python
class Scene5_ClosingTakeaway(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        summary = VGroup(
            Text("stable activations", font=MONO, font_size=28, color=P["PRIMARY"]),
            Text("smoother optimization", font=MONO, font_size=28, color=P["SECONDARY"]),
            Text("less brittle training", font=MONO, font_size=28, color=P["ACCENT"]),
        ).arrange(DOWN, buff=0.35)
        self.add(summary)
        self.wait(2)
```

Expected: the ending reinforces the core idea without adding a new concept.

- [ ] **Step 2: Render the full set of scenes**

```bash
manim -qh Blog-research/manim-batchnorm/script.py \
  Scene1_ActivationCloud \
  Scene2_BatchNormalize \
  Scene3_GammaBeta \
  Scene4_TrainVsInference \
  Scene5_ClosingTakeaway
```

Expected: production renders complete for all five scenes.

- [ ] **Step 3: Stitch the scenes if the final video is composed from clips**

```bash
ffmpeg -f concat -safe 0 -i concat.txt -c copy Blog-research/manim-batchnorm/final.mp4
```

Expected: one final MP4 that can be reviewed as a single explainer.

- [ ] **Step 4: Do a final visual pass**

```bash
manim -ql --format=png -s Blog-research/manim-batchnorm/script.py Scene2_BatchNormalize
```

Expected: at least one still frame is clean enough that the layout can be trusted before publishing.

---

## Self-Review Checklist

- Each scene has a single job.
- No scene title leaks into the frame unless it is doing real explanatory work.
- 3D scenes show geometry first, not labels first.
- `gamma` and `beta` are explained as affine recovery, not as a side note.
- Train vs inference uses different statistics explicitly.
- The final video fits the concept note without repeating the whole note verbatim.
