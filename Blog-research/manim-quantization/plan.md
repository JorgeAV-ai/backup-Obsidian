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

### Scene 1: Memory Wall
- misconception to correct: quantization is a minor numerical trick rather than a response to a hard memory constraint.
- aha moment: the same model shifts from impractical to manageable when its representation drops from FP16 to lower-bit formats.
- composition: one clean horizontal comparison of the same 70B model in FP16, INT8, and INT4 using direct scale bars and compact labels, with no hardware illustration.
- dominant color: coral for the expensive full-precision case, with amber and green signaling lower-bit relief.
- must show: relative footprint differences, concrete memory numbers, and the idea that bit-width changes the deployment budget materially.
- must avoid: decorative GPUs, extra side narratives about serving, and abstract arrows that do not teach anything.

### Scene 2: Quantization Intuition
- misconception to correct: quantization is magic compression rather than collapsing many continuous values into fewer discrete levels.
- aha moment: nearby values that were distinct in continuous space are forced into the same small set of quantized steps.
- composition: warm continuous points on the left, cool discrete bins or steps on the right, with clear visual mapping from one side into the other.
- dominant color: peach for the continuous side and sky blue for the discrete side.
- must show: the move from continuous variation to a smaller discrete vocabulary and the resulting loss of resolution.
- must avoid: overly metaphorical visuals, too much explanatory text, or a composition that shows motion without making the loss of granularity visible.

### Scene 3: Quantization Formula
- misconception to correct: the quantization formula is too abstract to support the rest of the article.
- aha moment: each symbol in the formula carries one specific job and the whole pipeline can be read visually from left to right.
- composition: the main quantization formula centered, a dequantization line beneath it, and a compact legend or annotation layer that reinforces the same semantic colors.
- dominant color: semantic rather than single-tone, with blue for W, green for Delta, amber for Z, red for Round plus Clip, and violet for W_q.
- must show: a readable sequence from original value to quantized value to approximate reconstruction, with error tied to step size rather than arbitrary noise.
- must avoid: decorative symbolism, too many colors, or changing semantic color meaning between the quantization and dequantization lines.

### Scene 4: Outlier Distortion
- misconception to correct: if the formula works on paper, applying it naively to a whole tensor should be enough.
- aha moment: one extreme outlier stretches the quantization scale and crushes the useful resolution of the ordinary values.
- composition: a normal cluster of values, one clearly separated outlier, and a coarse quantization grid or bucket view that makes the collapse of the center visually obvious.
- dominant color: calm blue for normal values and red for the outlier, with green reserved for the inflated Delta label if needed.
- must show: the causal link between outlier magnitude, larger step size, and loss of effective resolution in the bulk of the tensor.
- must avoid: generic histograms disconnected from the quantization grid, long text overlays, or a dramatic outlier without a visible mechanical consequence.
