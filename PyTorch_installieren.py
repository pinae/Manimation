from manim import Scene
from manim import Text, Code, Line, VGroup
from manim import Write, Unwrite, Create, FadeOut
from manim import UP, DOWN, LEFT, RIGHT


class ToDoAnimation(Scene):
    def construct(self):
        self.camera.background_color = '#261b50'
        font_settings = {"font": "Saira ExtraCondensed", "font_size": 95}
        s1 = Text("1.)  aktueller Grafiktreiber", **font_settings).move_to(2.5 * UP)
        s2 = Text("2.)  CUDA-Framework", **font_settings)
        s3 = Text("3.)  cuDNN", **font_settings)
        s4 = Text("4.)  Python und PyTorch", **font_settings)
        text_group = VGroup(s1, s2, s3, s4)
        text_group.arrange(DOWN, center=False, aligned_edge=LEFT)
        play_time = 15.0
        pip_command = Code(code="pip install pytorch", tab_width=4, language='python',
                           background='window', insert_line_no=False)
        self.wait(0.25)
        self.play(Create(pip_command), run_time=3.0)
        l1 = Line(4 * LEFT + 2 * DOWN, 4 * RIGHT + 2 * UP)
        l2 = Line(4 * LEFT + 2 * UP, 4 * RIGHT + 2 * DOWN)
        self.play(Write(l1), run_time=.25)
        self.play(Write(l2), run_time=.25)
        self.wait(3.25)
        self.play(FadeOut(pip_command, l1, l2), run_time=1.0)
        self.wait(5.0)
        self.play(Write(s1), run_time=(play_time-2.0)/4)
        self.wait(0.5)
        self.play(Write(s2), run_time=(play_time-2.0)/4)
        self.wait(0.5)
        self.play(Write(s3), run_time=(play_time-2.0)/4)
        self.wait(0.5)
        self.play(Write(s4), run_time=(play_time-2.0)/4)
        self.wait(2.5)
        self.play(FadeOut(s1, s2, s3, s4), run_time=1.0)
        self.wait(5.0)
