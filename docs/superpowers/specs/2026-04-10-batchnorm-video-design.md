# BatchNorm Video Design Spec

**Date**: 2026-04-10  
**Status**: Approved  
**Target**: `Blog-research/manim-batchnorm/`

---

## Goal

Create a short Manim explainer video that shows what BatchNorm does in a way that is visually intuitive first and mathematically exact second. The video should use a hybrid format: 3D to show the effect on activations, 2D overlays for the minimal formulas and train vs inference distinction.

---

## Scope

This video is for the BatchNorm concept note in `Basics/BatchNorm.md`. The video should help a reader understand:

- why activations become easier to train when their distribution is stabilized,
- how BatchNorm recenters and rescales a batch,
- what `gamma` and `beta` do,
- why training and inference use different statistics.

The video should not try to be a full lecture on normalization layers, optimization theory, or covariance shift history. It should stay compact and visual.

---

## Visual Direction

- **Style**: pastel, clean, restrained, slightly editorial
- **Format**: hybrid 3D + 2D
- **Background**: light warm tone, not dark
- **Text policy**: no scene titles, no long explanatory text inside the frame, no overexplaining labels
- **Typography**: minimal monospace labels only when needed
- **Emphasis**: geometry first, equations second

The 3D parts should feel like a physical transformation of points or blobs in space, not like a generic scatter plot. The 2D overlays should be sparse and only appear when they add precision.

---

## Narrative Arc

The video should progress from problem to mechanism to subtlety:

1. A batch of activations starts off offset and stretched.
2. BatchNorm recenters and rescales that batch into a more stable shape.
3. `gamma` and `beta` reintroduce learnable flexibility.
4. Training uses batch statistics; inference uses running statistics.

The viewer should leave with one clear mental model: BatchNorm is not "magic normalization", it is a controlled reparameterization that keeps activations well-behaved during training.

---

## Scene Plan

### Scene 1: Unstable activations

Show a 3D cloud or layered point field representing a mini-batch of activations with uneven mean and spread. The cloud should clearly look shifted away from center and stretched along one axis.

Purpose:

- establish the problem visually,
- make "different batch, different distribution" feel concrete.

### Scene 2: Normalize the batch

Transform the same 3D cloud so it recenters around zero and becomes more balanced in spread. Use one or two minimal 2D annotations for mean shift and scale reduction, but keep the focus on the geometric morph.

Purpose:

- show the core BatchNorm operation,
- make centering and variance normalization visible.

### Scene 3: Learnable scale and shift

Add a sparse 2D overlay showing `gamma` and `beta` as the knobs that can stretch and move the normalized batch back into a useful range. This should make it obvious that BatchNorm does not destroy representational power.

Purpose:

- explain why the normalization step does not constrain the model,
- show the affine recovery step.

### Scene 4: Training vs inference

Use a split visual: batch statistics on one side, running statistics on the other. The same batch should look stable in both cases, but the labels and arrows should make it clear that the source of statistics changes at inference time.

Purpose:

- communicate the most important subtlety,
- prevent a common misunderstanding that BatchNorm always uses the current batch.

### Scene 5: Closing takeaway

End with a compact visual summary that reinforces the training benefit: smoother distribution, steadier optimization, less sensitivity to initialization. This should be a closing image rather than a new concept.

Purpose:

- leave a single takeaway,
- avoid a text-heavy ending.

---

## File Map

| File | Role |
|---|---|
| `Blog-research/manim-batchnorm/script.py` | Manim scenes for the video |
| `Blog-research/manim-batchnorm/plan.md` | Scene-by-scene implementation notes |
| `docs/superpowers/specs/2026-04-10-batchnorm-video-design.md` | This design spec |

---

## Constraints

- Use one class per scene.
- Keep each scene independently renderable.
- Keep labels short and monospace.
- Use `self.wait()` after key reveals so the viewer can absorb the transformation.
- Do not put titles inside scenes unless absolutely necessary.
- Do not reuse the exact look of the quantization video; BatchNorm should have its own visual identity.

---

## Success Criteria

The video is successful if:

- a viewer can explain BatchNorm in one sentence after watching,
- the 3D scene makes the distribution shift obvious without text,
- the `gamma` / `beta` step is visually distinct from the normalization step,
- the train vs inference distinction is not ambiguous,
- the video looks intentional and clean on first render.

---

## Non-Goals

- Full derivation of the backward pass
- A comparison video against LayerNorm / InstanceNorm / GroupNorm
- A full lecture on internal covariate shift
- An animation with heavy narration or dense captions

---

## Open Questions Resolved

- **Pastel or dark?** Pastel. This is the chosen direction.
- **Pure 3D or hybrid?** Hybrid. 3D for intuition, 2D for the exact idea.
- **How much text?** Minimal, only to disambiguate the math and statistics.

