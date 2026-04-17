from manim import *
import numpy as np

MONO = "Menlo"
TITLE_SIZE = 42
BODY_SIZE = 28
LABEL_SIZE = 22
SMALL_SIZE = 18

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


def title_block(title, subtitle):
    title_text = Text(
        title,
        font=MONO,
        font_size=TITLE_SIZE,
        color=P["TEXT"],
        weight=BOLD,
    )
    subtitle_text = soften(
        Text(subtitle, font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]),
        0.45,
    )
    block = VGroup(title_text, subtitle_text).arrange(DOWN, buff=0.18)
    block.to_edge(UP, buff=0.5)
    return block


def memory_bar(label, gb, max_gb, color):
    bar_height = 3.9 * gb / max_gb
    bar = RoundedRectangle(
        corner_radius=0.12,
        width=1.55,
        height=max(bar_height, 0.08),
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.42,
    )
    value_text = Text(
        f"{gb} GB",
        font=MONO,
        font_size=LABEL_SIZE,
        color=color,
        weight=BOLD,
    )
    name_text = soften(
        Text(label, font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]),
        0.78,
    )
    value_text.next_to(bar, UP, buff=0.15)
    name_text.next_to(bar, DOWN, buff=0.22)
    return VGroup(bar, value_text, name_text)


class Scene1MemoryWall(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.add(title_block("The Memory Wall", "Same 70B model, different precision budgets"))

        labels = ["70B FP16", "70B INT8", "70B INT4"]
        values = [140, 70, 35]
        colors = [P["RED"], P["AMBER"], P["GREEN"]]
        max_gb = 160
        x_positions = [-3.8, 0, 3.8]
        base_y = -1.5

        bars = VGroup()
        for label, value, color, x in zip(labels, values, colors, x_positions):
            group = memory_bar(label, value, max_gb, color)
            group[0].move_to(np.array([x, base_y + group[0].height / 2, 0]))
            group[1].next_to(group[0], UP, buff=0.15)
            group[2].next_to(group[0], DOWN, buff=0.22)
            bars.add(group)

        baseline = soften(
            Line(np.array([-5.6, base_y, 0]), np.array([5.6, base_y, 0]), color=P["TEXT"], stroke_width=1.5),
            0.2,
        )
        guide = soften(
            DashedLine(
                np.array([-5.6, base_y + 3.9 * 80 / max_gb, 0]),
                np.array([5.6, base_y + 3.9 * 80 / max_gb, 0]),
                color=P["AMBER"],
                stroke_width=1.2,
                dash_length=0.16,
            ),
            0.35,
        )
        guide_label = soften(
            Text("80 GB class GPU memory", font=MONO, font_size=SMALL_SIZE, color=P["AMBER"]),
            0.55,
        )
        guide_label.next_to(guide, LEFT, buff=0.35).shift(UP * 0.18)

        note = soften(
            Text("Lower bit-width shrinks the memory footprint directly", font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]),
            0.48,
        )
        note.to_edge(DOWN, buff=0.65)

        self.add(baseline, guide, guide_label, bars, note)


class Scene2QuantizationIntuition(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.add(title_block("Continuous to Discrete", "Many nearby values must share fewer levels"))

        left_panel = RoundedRectangle(
            corner_radius=0.18,
            width=5.3,
            height=4.2,
            stroke_color=P["PEACH"],
            stroke_width=2,
            fill_color=P["PEACH"],
            fill_opacity=0.16,
        ).move_to(np.array([-3.3, -0.2, 0]))
        right_panel = RoundedRectangle(
            corner_radius=0.18,
            width=5.3,
            height=4.2,
            stroke_color=P["SKY"],
            stroke_width=2,
            fill_color=P["SKY"],
            fill_opacity=0.16,
        ).move_to(np.array([3.3, -0.2, 0]))

        left_points = VGroup(*[
            Dot(point=np.array([x, np.sin(x) * 0.35, 0]), radius=0.08, color=P["PEACH"])
            for x in np.linspace(-2.5, 2.5, 18)
        ])
        left_points.move_to(left_panel.get_center() + LEFT * 0.1)
        for i, dot in enumerate(left_points):
            dot.set_fill(interpolate_color(P["PEACH"], P["CORAL"], i / max(1, len(left_points) - 1)))

        levels_y = [-0.85, -0.25, 0.35, 0.95]
        steps = VGroup()
        for row, y in enumerate(levels_y):
            step = VGroup()
            row_width = 1.9 + 0.35 * row
            for col in range(4 if row < 3 else 5):
                rect = RoundedRectangle(
                    corner_radius=0.08,
                    width=row_width / (4 if row < 3 else 5),
                    height=0.28,
                    stroke_color=P["BLUE"],
                    stroke_width=1.8,
                    fill_color=interpolate_color(P["SKY"], P["VIOLET"], row / 3),
                    fill_opacity=0.25,
                )
                step.add(rect)
            step.arrange(RIGHT, buff=0.04)
            step.move_to(right_panel.get_center() + np.array([0.0, y, 0]))
            steps.add(step)

        arrow = Arrow(
            left_panel.get_right() + RIGHT * 0.2,
            right_panel.get_left() + LEFT * 0.2,
            color=P["VIOLET"],
            stroke_width=6,
            max_tip_length_to_length_ratio=0.12,
            buff=0.0,
        )

        left_label = soften(Text("continuous values", font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]), 0.55)
        right_label = soften(Text("shared quantized levels", font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]), 0.55)
        left_label.next_to(left_panel, DOWN, buff=0.22)
        right_label.next_to(right_panel, DOWN, buff=0.22)

        self.add(left_panel, right_panel, left_points, steps, arrow, left_label, right_label)


class Scene3QuantizationFormula(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.add(title_block("Uniform Quantization", "Each symbol plays one concrete role"))

        quant = MathTex(
            r"W_q",
            r"=",
            r"\operatorname{RoundClip}\left(",
            r"\frac{W}{\Delta}",
            r"+",
            r"Z",
            r"\right)",
            font_size=44,
            color=P["TEXT"],
        )
        quant[0].set_color(P["VIOLET"])
        quant[2].set_color(P["RED"])
        quant[3].set_color(P["TEXT"])
        quant[3][1].set_color(P["BLUE"])
        quant[3][3].set_color(P["GREEN"])
        quant[5].set_color(P["AMBER"])
        quant.move_to(UP * 1.3)

        dequant = MathTex(
            r"W_{\mathrm{recon}}",
            r"=",
            r"\left(",
            r"W_q",
            r"-",
            r"Z",
            r"\right)",
            r"\Delta",
            font_size=36,
            color=P["TEXT"],
        )
        dequant[0].set_color(P["TEXT"])
        dequant[3].set_color(P["VIOLET"])
        dequant[5].set_color(P["AMBER"])
        dequant[7].set_color(P["GREEN"])
        dequant.move_to(np.array([0, 0.25, 0]))

        error_note = soften(
            Text("larger Δ means a coarser grid and larger reconstruction error", font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]),
            0.48,
        )
        error_note.next_to(dequant, DOWN, buff=0.28)

        legend_items = [
            ("W", "original value", P["BLUE"]),
            ("\u0394", "step size", P["GREEN"]),
            ("Z", "zero-point", P["AMBER"]),
            ("Round + Clip", "map to discrete range", P["RED"]),
            ("W_q", "quantized value", P["VIOLET"]),
        ]
        legend = VGroup()
        for symbol, desc, color in legend_items:
            symbol_text = Text(symbol, font=MONO, font_size=LABEL_SIZE, color=color, weight=BOLD)
            dash = Text("  -  ", font=MONO, font_size=LABEL_SIZE - 2, color=P["TEXT"])
            desc_text = soften(Text(desc, font=MONO, font_size=LABEL_SIZE - 2, color=P["TEXT"]), 0.62)
            item = VGroup(symbol_text, dash, desc_text).arrange(RIGHT, buff=0.03)
            legend.add(item)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(np.array([0, -2.15, 0]))

        divider = soften(
            Line(np.array([-5.2, -0.55, 0]), np.array([5.2, -0.55, 0]), color=P["TEXT"], stroke_width=1.0),
            0.15,
        )

        self.add(quant, divider, dequant, error_note, legend)


class Scene4OutlierDistortion(Scene):
    def construct(self):
        self.camera.background_color = P["BG"]
        self.add(title_block("Why Naive Quantization Fails", "One outlier can ruin the useful resolution of the rest"))

        left_panel = RoundedRectangle(
            corner_radius=0.18,
            width=5.3,
            height=4.25,
            stroke_color=P["SKY"],
            stroke_width=2,
            fill_color=P["SKY"],
            fill_opacity=0.14,
        ).move_to(np.array([-3.3, -0.2, 0]))
        right_panel = RoundedRectangle(
            corner_radius=0.18,
            width=5.3,
            height=4.25,
            stroke_color=P["MINT"],
            stroke_width=2,
            fill_color=P["MINT"],
            fill_opacity=0.14,
        ).move_to(np.array([3.3, -0.2, 0]))

        normal_x = np.linspace(-2.0, 2.0, 17)
        normal_dots = VGroup(*[
            Dot(point=np.array([x, 0.22 * np.sin(2 * x), 0]), radius=0.08, color=P["BLUE"])
            for x in normal_x
        ])
        normal_dots.move_to(left_panel.get_center() + LEFT * 0.1 + DOWN * 0.15)
        outlier = Dot(point=left_panel.get_center() + np.array([1.95, 1.15, 0]), radius=0.11, color=P["RED"])
        outlier_label = soften(Text("outlier", font=MONO, font_size=SMALL_SIZE, color=P["RED"]), 0.8)
        outlier_label.next_to(outlier, UP, buff=0.15)

        grid_x = np.linspace(-1.75, 1.75, 5)
        coarse_levels = [-0.95, -0.25, 0.45, 1.15]
        buckets = VGroup()
        for y in coarse_levels:
            row = VGroup()
            for x in grid_x:
                rect = RoundedRectangle(
                    corner_radius=0.06,
                    width=0.68,
                    height=0.26,
                    stroke_color=P["GREEN"],
                    stroke_width=1.6,
                    fill_color=P["MINT"],
                    fill_opacity=0.18,
                )
                rect.move_to(right_panel.get_center() + np.array([x, y, 0]))
                row.add(rect)
            buckets.add(row)

        collapsed = VGroup(*[
            Dot(point=right_panel.get_center() + np.array([x, y, 0]), radius=0.075, color=P["BLUE"])
            for x, y in [
                (-1.45, -0.25), (-0.7, -0.25), (0.0, -0.25), (0.7, -0.25), (1.4, -0.25),
                (-1.1, 0.45), (-0.35, 0.45), (0.35, 0.45), (1.05, 0.45),
                (1.45, 1.15),
            ]
        ])

        arrow = Arrow(
            left_panel.get_right() + RIGHT * 0.2,
            right_panel.get_left() + LEFT * 0.2,
            color=P["RED"],
            stroke_width=6,
            max_tip_length_to_length_ratio=0.12,
            buff=0.0,
        )
        delta_label = soften(Text("larger \u0394", font=MONO, font_size=LABEL_SIZE, color=P["GREEN"]), 0.82)
        delta_label.next_to(arrow, UP, buff=0.18)

        left_label = soften(Text("normal values + one extreme outlier", font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]), 0.55)
        right_label = soften(Text("coarse grid collapses useful detail", font=MONO, font_size=SMALL_SIZE, color=P["TEXT"]), 0.55)
        left_label.next_to(left_panel, DOWN, buff=0.22)
        right_label.next_to(right_panel, DOWN, buff=0.22)

        self.add(
            left_panel,
            right_panel,
            normal_dots,
            outlier,
            outlier_label,
            buckets,
            collapsed,
            arrow,
            delta_label,
            left_label,
            right_label,
        )
