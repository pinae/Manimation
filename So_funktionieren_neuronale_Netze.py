from manim import Scene, ThreeDScene
from manim import Code, Text, MathTex, SVGMobject, Circle, Arrow, Arrow3D
from manim import Axes, ThreeDAxes, VGroup, Surface
from manim import FadeIn, Create, GrowArrow, DrawBorderThenFill, Write
from manim import Transform, TransformMatchingTex, MoveToTarget
from manim import FadeOut, Unwrite, Uncreate
from manim import UP, DOWN, LEFT, RIGHT, BLACK, RED, GREEN, BLUE, ORANGE, YELLOW, DEGREES, ORIGIN
from manim import config as global_config
import numpy as np


class NeuronAnimation(Scene):
    def construct(self):
        self.camera.background_color = '#404040'
        bio_neuron = SVGMobject(file_name="/home/pina/Dokumente/Artikel/Neuronale Netze erklärt/BioNeuron.svg",
                                should_center=True, height=7, fill_opacity=1.0, stroke_width=1.5)
        self.play(DrawBorderThenFill(bio_neuron), run_time=2.0)
        self.wait(duration=1)
        bio_neuron.save_state()
        self.play(bio_neuron.animate.set_color(RED), run_time=0.05)
        self.play(bio_neuron.animate.restore(), run_time=0.2)
        self.wait(duration=1)
        self.play(bio_neuron.animate.set_color(RED), run_time=0.05)
        self.play(bio_neuron.animate.restore(), run_time=0.2)
        self.wait(duration=.1)
        self.play(bio_neuron.animate.set_color(RED), run_time=0.05)
        self.play(bio_neuron.animate.restore(), run_time=0.2)
        self.wait(duration=.1)
        self.play(bio_neuron.animate.set_color(RED), run_time=0.05)
        self.play(bio_neuron.animate.restore(), run_time=0.2)
        self.wait(duration=1)
        bio_neuron_kernlos = SVGMobject(
            file_name="/home/pina/Dokumente/Artikel/Neuronale Netze erklärt/BioNeuron_kernlos.svg",
            should_center=True, height=7, fill_opacity=1.0, stroke_width=1.5)
        self.play(FadeIn(bio_neuron_kernlos), run_time=1.0)
        self.play(FadeOut(bio_neuron), run_time=0.5)
        ki_neuron = SVGMobject(file_name="/home/pina/Dokumente/Artikel/Neuronale Netze erklärt/KI-Neuron.svg",
                               should_center=True, height=7, fill_opacity=1.0, stroke_width=1.5)
        self.play(Transform(bio_neuron_kernlos, ki_neuron, replace_mobject_with_target_in_scene=True), run_time=4.0)
        self.wait(duration=1)
        self.play(ki_neuron.animate.move_to(3 * RIGHT), ki_neuron.animate.scale(.6), run_time=1.0)
        i_n1 = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK).next_to(ki_neuron, LEFT)
        i_n1.move_to(6 * LEFT)
        i_n2 = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK).next_to(i_n1, UP)
        i_n3 = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK).next_to(i_n1, DOWN)
        self.play(Create(i_n1), Create(i_n2), Create(i_n3), run_time=1.0)
        a_i_n1 = Arrow(i_n1, ki_neuron.get_center() + 3.5 * LEFT, stroke_width=3)
        a_i_n2 = Arrow(i_n2, ki_neuron.get_center() + 3.5 * LEFT + 0.5 * UP, stroke_width=3)
        a_i_n3 = Arrow(i_n3, ki_neuron.get_center() + 3.5 * LEFT + 0.5 * DOWN, stroke_width=3)
        self.play(GrowArrow(a_i_n1), GrowArrow(a_i_n2), GrowArrow(a_i_n3), run_time=1.0)
        n = Circle(radius=2, fill_color="#e5ff80", fill_opacity=1.0,
                   stroke_width=1.5, stroke_color=BLACK).move_to(1.2 * RIGHT)
        move_vector = 3 * RIGHT
        self.play(Transform(ki_neuron, n, replace_mobject_with_target_in_scene=True),
                  i_n1.animate.move_to(i_n1.get_center() + move_vector),
                  a_i_n1.animate.move_to(a_i_n1.get_center() + move_vector),
                  i_n2.animate.move_to(i_n2.get_center() + move_vector),
                  a_i_n2.animate.move_to(a_i_n2.get_center() + move_vector),
                  i_n3.animate.move_to(i_n3.get_center() + move_vector),
                  a_i_n3.animate.move_to(a_i_n3.get_center() + move_vector),
                  run_time=0.5)
        self.wait(duration=.5)
        move_vector = 2 * UP
        self.play(n.animate.move_to(n.get_center() + move_vector),
                  i_n1.animate.move_to(i_n1.get_center() + move_vector),
                  a_i_n1.animate.move_to(a_i_n1.get_center() + move_vector),
                  i_n2.animate.move_to(i_n2.get_center() + move_vector),
                  a_i_n2.animate.move_to(a_i_n2.get_center() + move_vector),
                  i_n3.animate.move_to(i_n3.get_center() + move_vector),
                  a_i_n3.animate.move_to(a_i_n3.get_center() + move_vector),
                  run_time=1)
        formula = MathTex(r"\sum", r"x", font_size=72).next_to(n, DOWN)
        self.play(Write(formula), run_time=1)
        self.wait(duration=1)
        formula2 = MathTex(r"\sum", r"w", r"\cdot", r"x", font_size=72).move_to(formula)
        self.play(TransformMatchingTex(formula, formula2),
                  a_i_n1.animate.set(stroke_width=.75),
                  a_i_n2.animate.set(stroke_width=5),
                  a_i_n3.animate.set(stroke_width=8.5), run_time=2.0)
        self.wait(duration=.5)
        formula3 = MathTex(r"\sum_{i=0}^{n}", r"w", r"\cdot", r"x", font_size=72).next_to(n, DOWN)
        self.play(TransformMatchingTex(formula2, formula3), run_time=1.0)
        formula4 = MathTex(r"\sum_{i=0}^{n}", r"w_{i}", r"\cdot", r"x_{i}", font_size=72).move_to(formula3)
        self.play(TransformMatchingTex(formula3, formula4), run_time=1.0)
        self.wait(duration=1)
        formula5 = MathTex(r"\sum_{i=0}^{n}", r"w_{i}", r"\cdot", r"x_{i}", r"+ b", font_size=72).move_to(formula4)
        self.play(TransformMatchingTex(formula4, formula5), run_time=1.0)
        self.wait(duration=1)
        self.play(Uncreate(n),
                  Uncreate(a_i_n1), Uncreate(a_i_n2), Uncreate(a_i_n3),
                  Uncreate(i_n1), Uncreate(i_n2), Uncreate(i_n3),
                  Unwrite(formula5), run_time=1.5)
        self.wait(duration=.5)
        axes = Axes(
            x_range=[-1.0, 1.0, 0.1],
            y_range=[-1.2, 1.2, 0.1],
            x_length=10,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(-1.01, 1.01, .25),
                "numbers_with_elongated_ticks": np.arange(-1.0, 1.0, .5),
            },
            y_axis_config={
                "numbers_to_include": np.array([-1, -0.5, 0.5, 1.0]),
                "numbers_with_elongated_ticks": np.arange(-1.0, 1.0, .5),
            },
            tips=False,
        )
        axes_labels = axes.get_axis_labels()
        relu_graph = axes.plot(lambda x: np.maximum(0.0, x), color=BLUE)
        relu_graph.stroke_width = 7
        plot = VGroup(axes, relu_graph)
        labels = VGroup(axes_labels)
        self.play(Create(axes), Write(labels), run_time=2.0)
        self.play(Write(relu_graph), run_time=5.0)
        self.wait(duration=1)
        self.play(axes.animate.move_to(axes.get_center() + 2 * UP),
                  relu_graph.animate.move_to(relu_graph.get_center() + 2 * UP),
                  axes_labels.animate.move_to(axes_labels.get_center() + 2 * UP),
                  run_time=1.0)
        relu_code = Code(code="max(0, x)", language='python', background='window', insert_line_no=False)
        relu_code.next_to(plot, DOWN)
        self.play(Create(relu_code), run_time=1.0)
        self.wait(duration=2.0)
        self.play(Unwrite(plot), Unwrite(axes_labels),
                  relu_code.animate.move_to(relu_code.get_center() + 3 * UP), run_time=1.5)
        formula5 = MathTex(r"\sum_{i=0}^{n}", r"w_{i}", r"\cdot", r"x_{i}", r"+ b",
                           font_size=72).next_to(relu_code, DOWN)
        self.play(Write(formula5), run_time=1.0)
        formula6 = MathTex(r"max(0, " + r"\sum_{i=0}^{n}", r"w_{i}", r"\cdot", r"x_{i}", r"+ b", r")",
                           font_size=72).move_to(formula5)
        self.play(TransformMatchingTex(formula5, formula6), run_time=1.0)
        self.wait(duration=1.0)
        self.play(relu_code.animate.move_to(relu_code.get_center() + 10 * UP),
                  formula6.animate.move_to(formula6.get_center() + 10 * UP),
                  run_time=1.0)
        n0_1 = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK)
        n0_1.move_to(n0_1.get_center() + 2 * UP + 2 * LEFT)
        n0_2 = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK).next_to(n0_1, DOWN)
        n0_3 = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK).next_to(n0_2, DOWN)
        n1_1 = Circle(radius=2, fill_color="#e5ff80", fill_opacity=1.0,
                      stroke_width=1.5, stroke_color=BLACK).next_to(n0_2, RIGHT)
        n1_1.move_to(n1_1.get_center() + 2 * RIGHT)
        self.play(
            Create(n0_1), Create(n0_2), Create(n0_3),
            Create(n1_1),
            run_time=1.0)
        a1_1 = Arrow(n0_1, n1_1.get_center() + 1.8 * LEFT + UP, tip_length=.1)
        a1_2 = Arrow(n0_2, n1_1.get_center() + 1.8 * LEFT, tip_length=.1)
        a1_3 = Arrow(n0_3, n1_1.get_center() + 1.8 * LEFT + DOWN, tip_length=.1)
        self.play(
            GrowArrow(a1_1), GrowArrow(a1_2), GrowArrow(a1_3),
            run_time=.5)
        self.play(n1_1.animate.scale(.25),
                  a1_1.animate.put_start_and_end_on(a1_1.get_start(), n1_1.get_center() + 0.8 * LEFT + 0.2 * UP),
                  a1_2.animate.put_start_and_end_on(a1_2.get_start(), n1_1.get_center() + 0.8 * LEFT),
                  a1_3.animate.put_start_and_end_on(a1_3.get_start(), n1_1.get_center() + 0.8 * LEFT + 0.2 * DOWN),
                  run_time=2.0)
        self.wait(duration=.5)
        self.play(n0_1.animate.move_to(3 * DOWN + 6 * LEFT),
                  n0_2.animate.move_to(3 * DOWN + 5 * LEFT),
                  n0_3.animate.move_to(3 * DOWN + 4 * LEFT),
                  n1_1.animate.move_to(2 * DOWN + 5 * LEFT),
                  a1_1.animate.put_start_and_end_on(2.8 * DOWN + (6 - 0.2) * LEFT, 2.2 * DOWN + 5.2 * LEFT),
                  a1_2.animate.put_start_and_end_on(2.75 * DOWN + (5 + 0.0) * LEFT, 2.2 * DOWN + 5 * LEFT),
                  a1_3.animate.put_start_and_end_on(2.8 * DOWN + (4 + 0.2) * LEFT, 2.2 * DOWN + 4.8 * LEFT),
                  run_time=2.0)
        self.play(n0_1.animate.scale(.5),
                  n0_2.animate.scale(.5),
                  n0_3.animate.scale(.5),
                  n1_1.animate.scale(.5),
                  run_time=1.0)
        self.play(a1_1.animate.set_stroke(width=2),
                  a1_2.animate.set_stroke(width=2),
                  a1_3.animate.set_stroke(width=2),
                  un_time=1.0)
        neuron_counts = [10, 11, 12, 12, 10]
        layers = [[n0_1, n0_2, n0_3], [n1_1]]
        arrows = [[a1_1, a1_2, a1_3]]
        while len(layers) < len(neuron_counts) + 2:
            layers.append([])
        for i in range(10):
            new_n = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                           stroke_width=1.5, stroke_color=BLACK).scale(.5).move_to(3 * DOWN + (3 - i) * LEFT)
            layers[0].append(new_n)
            new_a = Arrow(new_n.get_center(),
                          n1_1.get_center(),
                          tip_length=.1, stroke_width=2)
            arrows[0].append(new_a)
            self.play(Create(new_n),
                      GrowArrow(new_a),
                      run_time=.1)
        for i in range(10):
            new_n = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                           stroke_width=1.5, stroke_color=BLACK).scale(.5).move_to(2 * DOWN + (4 - i) * LEFT)
            layers[1].append(new_n)
            new_layer1_arrows = []
            for lown in layers[0]:
                new_a = Arrow(lown.get_center(),
                              new_n.get_center(),
                              tip_length=.1, stroke_width=2)
                new_layer1_arrows.append(new_a)
            self.play(Create(new_n), *[GrowArrow(a) for a in new_layer1_arrows], run_time=.09)
            arrows[0] += new_layer1_arrows
        self.wait(duration=.5)
        for layer_index, neuron_count in enumerate(neuron_counts):
            for i in range(neuron_count):
                new_n = Circle(radius=.5, fill_color="#e5ff80", fill_opacity=1.0,
                               stroke_width=1.5, stroke_color=BLACK).scale(.5).move_to(
                    1 * DOWN + layer_index * UP + ((neuron_count * 0.5 - 0.5) - i) * LEFT)
                layers[2 + layer_index].append(new_n)
                new_arrows = []
                for lown in layers[1 + layer_index]:
                    new_a = Arrow(lown.get_center(),
                                  new_n.get_center(),
                                  tip_length=.1, stroke_width=2)
                    new_arrows.append(new_a)
                self.play(Create(new_n), *[GrowArrow(a) for a in new_arrows], run_time=.09)
                arrows.append(new_arrows)
                self.wait(duration=.03)
        self.wait(duration=1.0)


class LossAnimation(Scene):
    def construct(self):
        self.camera.background_color = '#404040'
        loss_formula = MathTex(r"Loss(", r"y', y", r") = ", r"(", r"y'", r" - ", r"y", r")^{2}")
        self.wait(duration=0.5)
        self.play(Write(loss_formula), run_time=3.0)
        self.wait(duration=1.0)
        loss_formula_sum = MathTex(r"Loss(", r"\vec{y'}, \vec{y}", r") = ", r'\sum_{i=0}^{n}', r"(", r"y'_{i}", r" - ",
                                   r"y_{i}", r")^{2}")
        self.play(TransformMatchingTex(loss_formula, loss_formula_sum), run_time=1.0)
        self.wait(duration=1.0)


class Loss3DPlotAnimation(ThreeDScene):
    def construct(self):
        # global_config.disable_caching = True
        self.camera.background_color = '#404040'
        resolution_fa = 24 * 2
        self.set_camera_orientation(phi=75 * DEGREES, theta=-80 * DEGREES, frame_center=1 * UP)
        self.begin_3dillusion_camera_rotation(rate=3.0)
        dataset = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 0],
                            [1, 20, 0],
                            [1, 20, 0],
                            [1, 0, 1],
                            [1, 0, 1],
                            [1, 0, 1]], dtype=np.float32)
        axes = ThreeDAxes(x_range=(-2, +2, .1), y_range=(-0.1, +0.1, .01), z_range=(0, 1, 0.1))
        labels = axes.get_axis_labels(
            MathTex(r"w_{1}").scale(1.5), MathTex(r"w_{2}").scale(2), MathTex(r"Loss(w_{1}, w_{2})").scale(1)
        )

        def loss(w1, w2):
            b = 0.0
            net = lambda x: np.maximum(w1 * x[0] + w2 * x[1] + b, 0)
            y_ = np.vectorize(net, signature='(n)->()')(dataset[:, 0:2])
            y = dataset[:, 2]
            return np.sum(np.power(y_ - y, 2)) / len(dataset)

        loss_plane = axes.plot_surface(
            loss,
            resolution=(resolution_fa, resolution_fa),
            u_range=[-2, +2],
            v_range=[-0.1, +0.1],
            colorscale=[BLUE, GREEN, YELLOW, ORANGE, RED]
        )

        self.play(Write(axes), Write(labels), run_time=2.0)
        self.play(Create(loss_plane), run_time=10.0)
        self.wait(2.0)

        def pp(x, y):
            return axes.c2p(x, y, loss(x, y))

        arr = Arrow3D(pp(0.8, 0.07),
                      pp(.05, -0.005),
                      color=RED)
        self.play(Create(arr), run_time=0.5)
        self.wait(2.5)
        arr2 = Arrow3D(pp(.05, -0.005),
                       pp(1.15, -0.02),
                       color=RED)
        self.play(Create(arr2), run_time=0.5)
        self.wait(2)
        arr3 = Arrow3D(pp(1.15, -0.02),
                       pp(1.0, -0.05),
                       color=RED)
        self.play(Create(arr3), run_time=0.5)
        self.wait(1.5)
        arr4 = Arrow3D(pp(1.0, -0.05),
                       pp(.78, -0.048),
                       color=RED)
        self.play(Create(arr4), run_time=0.5)
        self.wait(6.0)

        self.play(Uncreate(loss_plane), Uncreate(arr4), Uncreate(arr3), Uncreate(arr2), Uncreate(arr),
                  run_time=2.0)
        self.play(Unwrite(axes), Unwrite(labels), run_time=2.0)
        self.stop_3dillusion_camera_rotation()
        self.wait(duration=1.0)
