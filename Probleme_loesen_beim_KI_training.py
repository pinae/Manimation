from manim import Scene
from manim import Text, VGroup, Group
from manim import Write, Unwrite, Create, Uncreate, GrowArrow
from manim import Axes, CurvedArrow, Circle, Arrow
from manim import UP, DOWN, LEFT, RIGHT
from manim import BLUE, YELLOW, BLACK
from random import random

class ThreeBulletpoints(Scene):
    def construct(self):
        self.camera.background_color = '#404040'
        font_settings = {"font": "Saira ExtraCondensed", "font_size": 95}
        s1 = Text("1.)  Datensatz", **font_settings).move_to(1.5 * UP)
        s2 = Text("2.)  Netzwerkstruktur", **font_settings)
        s3 = Text("3.)  Hyperparameter", **font_settings)
        text_group = VGroup(s1, s2, s3)
        text_group.arrange(DOWN, center=False, aligned_edge=LEFT)
        self.play(Write(s1), run_time=.5)
        self.wait(1.5)
        self.play(Write(s2), run_time=.5)
        self.wait(0.5)
        self.play(Write(s3), run_time=.5)
        self.wait(2.5)

class Loss2D(Scene):
    def construct(self):
        self.camera.background_color = '#404040'

        def make_axes(x_max=10):
            axes = Axes(x_range=(0, x_max, 10), y_range=(0, 3), tips=False)
            axes.add_coordinates()
            return axes

        axes = make_axes(10)

        name = Text("Loss").next_to(axes, UP, buff=0)

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1.37, 1.34, 1.31, 1.25, 1.22, 1.15, 1.12, 1.07, 1.05, 1.02, 1.0]
        def make_graph(x, y, axes):
            graph = axes.plot_line_graph(x, y, add_vertex_dots=False, line_color=BLUE)
            return graph

        graph = make_graph(x, y, axes)
        self.play(Write(axes), Write(name), run_time=1)
        self.play(Write(graph), run_time=1)
        self.wait(1)

        def make_step_arrow(s_c, e_c, axes, angle=-2.0):
            return CurvedArrow(axes.c2p(*s_c), axes.c2p(*e_c), angle=angle, color=YELLOW)

        small_step = make_step_arrow((2, 1.31), (3, 1.25), axes)
        self.play(Write(small_step), run_time=0.5)

        for i in range(3, 7):
            self.remove(small_step)
            small_step = make_step_arrow((2, 1.31), (x[i], y[i]), axes)
            self.add(small_step)
            self.wait(1/25)
        self.wait(.5)
        for i in range(7, 3, -1):
            self.remove(small_step)
            small_step = make_step_arrow((2, 1.31), (x[i], y[i]), axes)
            self.add(small_step)
            self.wait(1/25)


        #big_step = make_step_arrow((2, 1.31), (17, 2.44), axes, angle=-0.5)
        #self.play(Write(big_step), run_time=0.5)
        #self.wait(1)

        #x_additions = list(range(11, 21, 1))
        #y_additions = [1.1, 1.5, 1.85, 2.17, 2.42, 2.45, 2.44, 2.23, 2.12, 2.48]
        #for i, x_a in enumerate(x_additions):
        #    x.append(x_a)
        #    y.append(y_additions[i])
        #    new_axes = make_axes(x[-1])
        #    new_graph = make_graph(x, y, new_axes)
            #self.remove(small_step, big_step)
            #small_step = make_step_arrow((2, 1.31), (3, 1.25), new_axes)
            #big_step = make_step_arrow((2, 1.31), (17, 2.44), new_axes, angle=-0.5)
            #self.add(small_step, big_step)
        #    self.add(axes.become(new_axes), graph.become(new_graph))
        #    self.wait(.1)
        self.wait(1)


class NeuronAnimation(Scene):
    def construct(self):
        self.camera.background_color = '#404040'
        neuron_layers = []
        for layer_no in range(3):
            neuron_layers.append([])
            for i in range(10):
                neuron_layers[-1].append(
                    Circle(radius=.5, fill_color="#e5ff80", fill_opacity=random(),
                           stroke_width=1.5, stroke_color=BLACK)
                )
            layer_group = Group(*neuron_layers[layer_no])
            layer_group.arrange(RIGHT, center=True, aligned_edge=DOWN).move_to(3*DOWN + layer_no*3*UP)
        self.play([Create(c) for c in neuron_layers[0]], run_time=.8)
        self.play([Create(c) for c in neuron_layers[1]], run_time=.8)
        self.play([Create(c) for c in neuron_layers[2]], run_time=.8)

        weight_arrows = []
        for i, neuron_layer in enumerate(neuron_layers):
            if i >= len(neuron_layers)-1:
                break
            weight_arrows.append([])
            for source_neuron in neuron_layer:
                for target_neuron in neuron_layers[i+1]:
                    weight_arrows[i].append(
                        Arrow(source_neuron.get_center(), target_neuron.get_center(), stroke_width=10*random())
                    )
        self.play([GrowArrow(arr) for arrl in weight_arrows for arr in arrl], run_time=1.0)
        self.wait(1)
        #a = Arrow(source_neuron.get_center(), target_neuron.get_center(), stroke_width=10*random())
        #a.animate.set_stroke(width=0)
        #n:Circle = neuron_layers[2][0]
        #n.animate.set_fill(opacity=0)
        self.play([a.animate.set_stroke(width=0.1) for a in weight_arrows[1]],
                  [n.animate.set_fill(opacity=1) if i == 3 else n.animate.set_fill(opacity=0) for i, n in enumerate(neuron_layers[2])],
                  run_time=4)
        self.wait(3)
