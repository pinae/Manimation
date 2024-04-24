from manim import Code, Text, MathTex
from manim import MoveToTarget, FadeOut, Create, Transform, Write, Unwrite, FadeIn
from manim import UP, DOWN, LEFT, RIGHT
from manim import Scene, BarChart, TransformMatchingTex
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService


def make_code(code_str: str, line_no_from: int = 1) -> Code:
    new_code_object = Code(code=code_str, tab_width=4, language='python',
                           background='window', line_no_from=line_no_from)
    new_code_object.line_numbers.color = '#707070'
    return new_code_object

class CodeAnimation(VoiceoverScene):
    def construct(self):
        #self.set_speech_service(RecorderService())
        self.camera.background_color = '#261b50'
        code = '''import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(
    training_data, batch_size=batch_size)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FeedForwardNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=1e-3)
        '''
        # file_name='../PyTorch-MNIST-Example/train-MNIST.py'
        full_code = make_code(code)
        full_code.shift(8 * DOWN)
        #full_code = Text("Foo bar.")
        self.add(full_code)
        #with self.voiceover(text="Ich habe für euch DAS Standard Beispiel dafür rausgesucht.") as tracker:
        full_code.generate_target()
        full_code.target.shift(17 * UP)
        self.play(MoveToTarget(full_code), run_time=2)
        self.play(FadeOut(full_code), run_time=0.5)
        self.wait(duration=1)

#         code = '''import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor'''
#         c1 = make_code(code)
#         self.play(Create(c1), run_time=2)
#         self.wait(duration=2)
#
#         code = '''training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )'''
#         c2 = make_code(code, 7)
#         #c2.next_to(c1, direction=DOWN)
#         c1.generate_target()
#         c1.target.next_to(c2, direction=UP)
#         self.play(MoveToTarget(c1), run_time=1)
#         self.play(Create(c2), run_time=2)
#         self.wait(duration=2)
#         c2.code += '''
# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )'''
#         self.play(c2.animate, run_time=2)
#         self.wait(duration=2)

class CrossEntropyAnimation(Scene):
    def construct(self):
        self.camera.background_color = '#404040'
        yis3 = MathTex("y=4", font_size=72)
        self.play(Write(yis3), run_time=3.0)
        self.wait(4.0)
        one_hot = MathTex(r"(0.0, ", r"0.0, ", r"0.0, ", r"0.0, ", r"1.0, ", r"0.0, ", r"0.0, ", r"0.0, ", r"0.0, ", r"0.0)", font_size=66).move_to(0.3 * RIGHT)
        self.play(Unwrite(yis3), Write(one_hot), run_time=1.0)
        self.wait(1.0)
        self.play(one_hot.animate.move_to(2*DOWN + 0.3*RIGHT), run_time=1.0)
        self.wait(0.3)
        one_hot_chart = BarChart(
            values=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            bar_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            y_range=[0, 1, .2],
            y_length=3,
            x_length=12,
            x_axis_config={"font_size": 36},
        ).move_to(UP)
        one_hot_c_bar_lbls = one_hot_chart.get_bar_labels(font_size=48)
        self.play(Write(one_hot_chart), Write(one_hot_c_bar_lbls), run_time=3.0)
        self.wait(5.0)
        self.play(one_hot.animate.move_to(3*DOWN + 0.3*RIGHT), run_time=1.0)
        logits = MathTex(r"(0.0, ", r"0.0, ", r"0.2, ", r"0.1, ", r"0.6, ", r"0.0, ", r"0.1, ", r"0.0, ", r"0.0, ", r"0.0)", font_size=66).move_to(2*DOWN + 0.3*RIGHT)
        #self.play(Write(logits), run_time=1.0)
        #self.wait(1.0)
        #self.play(one_hot.animate.move_to(3*DOWN + 0.3*RIGHT), logits.animate.move_to(2*DOWN + 0.3*RIGHT), run_time=1.0)
        chart = BarChart(
            values=[0, 0, 0.2, 0.1, 0.6, 0, 0.1, 0, 0, 0],
            bar_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            y_range=[0, 1, .2],
            y_length=3,
            x_length=12,
            x_axis_config={"font_size": 36},
        ).move_to(UP)
        c_bar_lbls = chart.get_bar_labels(font_size=48)
        self.play(FadeOut(one_hot_chart), FadeOut(one_hot_c_bar_lbls), Write(logits), FadeIn(chart), FadeIn(c_bar_lbls), run_time=3.0)
        self.wait(3.0)
        logits2 = MathTex(r"(0.0, ", r"0.0, ", r"0.0, ", r"0.0, ", r"0.5, ", r"0.0, ", r"0.0, ", r"0.0, ", r"0.0, ",
                         r"0.5)", font_size=66).move_to(2 * DOWN + 0.3 * RIGHT)
        chart2 = BarChart(
            values=[0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5],
            bar_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            y_range=[0, 1, .2],
            y_length=3,
            x_length=12,
            x_axis_config={"font_size": 36},
        ).move_to(UP)
        c_bar_lbls2 = chart2.get_bar_labels(font_size=48)
        self.play(FadeOut(chart), FadeOut(c_bar_lbls), Unwrite(logits), Write(logits2), FadeIn(chart2), FadeIn(c_bar_lbls2),
                  run_time=0.5)
        self.wait(2.0)
        self.play(Unwrite(one_hot), Unwrite(logits2), Unwrite(chart2), Unwrite(c_bar_lbls2), run_time=1.0)
        self.wait(1.0)
        full_formula = MathTex(r"H(", r"P", r", ", r"Q", r") = ", r"-", r"\sum_{i=0}^{10}", r"P_{i}", r"\cdot",
                               r"\log Q_{i}", r" ", r" ", font_size=72)
        self.play(Write(full_formula), run_time=1.0)
        self.wait(1.0)
        both0 = MathTex(r"H(", r"0", r", ", r"0", r") = ", r"-", r"\sum_{i=0}^{10}", r"0", r"\cdot", r"\log 0",
                        r" = ", r"0",
                        font_size=72)
        self.play(Transform(full_formula.submobjects[0], both0.submobjects[0],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[1], both0.submobjects[1],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[2], both0.submobjects[2],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[3], both0.submobjects[3],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[4], both0.submobjects[4],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[5], both0.submobjects[5],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[6], both0.submobjects[6],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[7], both0.submobjects[7],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[8], both0.submobjects[8],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[9], both0.submobjects[9],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[10], both0.submobjects[10],
                            replace_mobject_with_target_in_scene=False),
                  Transform(full_formula.submobjects[11], both0.submobjects[11],
                            replace_mobject_with_target_in_scene=False),
                  run_time=1.0)
        self.play(TransformMatchingTex(full_formula, both0, fade_transform_mismatches=True), run_time=0.001)
        self.wait(1.0)
        both1 = MathTex(r"H(", r"1", r", ", r"1", r") = ", r"-", r"\sum_{i=0}^{10}", r"1", r"\cdot", r"\log 1",
                        r" = ", r"0",
                        font_size=72)
        self.play(Transform(both0.submobjects[1], both1.submobjects[1], replace_mobject_with_target_in_scene=False),
                  Transform(both0.submobjects[3], both1.submobjects[3], replace_mobject_with_target_in_scene=False),
                  Transform(both0.submobjects[7], both1.submobjects[7], replace_mobject_with_target_in_scene=False),
                  Transform(both0.submobjects[9], both1.submobjects[9], replace_mobject_with_target_in_scene=False),
                  #FadeOut(both0),
                  run_time=1.0)
        self.play(TransformMatchingTex(both0, both1, transform_mismatches=True), run_time=0.001)
        self.wait(1.0)
        p1q0_ = MathTex(r"H(", r"1", r", ", r"0", r") = ", r" ", r"\sum_{i=0}^{10}", r"1", r"\cdot", r"\log 0",
                        r" = ", r"-\infty",
                        font_size=72)
        self.play(#TransformMatchingTex(both1, p1q0_, transform_mismatches=True),
            Transform(both1.submobjects[0], p1q0_.submobjects[0], replace_mobject_with_target_in_scene=False),
                  Transform(both1.submobjects[1], p1q0_.submobjects[1], replace_mobject_with_target_in_scene=False),
            Transform(both1.submobjects[2], p1q0_.submobjects[2], replace_mobject_with_target_in_scene=False),
                  Transform(both1.submobjects[3], p1q0_.submobjects[3], replace_mobject_with_target_in_scene=False),
            Transform(both1.submobjects[4], p1q0_.submobjects[4], replace_mobject_with_target_in_scene=False),
                  Transform(both1.submobjects[5], p1q0_.submobjects[5], replace_mobject_with_target_in_scene=False),
            Transform(both1.submobjects[6], p1q0_.submobjects[6], replace_mobject_with_target_in_scene=False),
                  Transform(both1.submobjects[7], p1q0_.submobjects[7], replace_mobject_with_target_in_scene=False),
            Transform(both1.submobjects[8], p1q0_.submobjects[8], replace_mobject_with_target_in_scene=False),
                  Transform(both1.submobjects[9], p1q0_.submobjects[9], replace_mobject_with_target_in_scene=False),
            Transform(both1.submobjects[10], p1q0_.submobjects[10], replace_mobject_with_target_in_scene=False),
                  Transform(both1.submobjects[11], p1q0_.submobjects[11], replace_mobject_with_target_in_scene=False),
            #FadeOut(both1),
                  run_time=1.0)
        self.play(TransformMatchingTex(both1, p1q0_, transform_mismatches=True), run_time=0.001)
        self.wait(1.0)
        p1q0 = MathTex(r"H(", r"1", r", ", r"0", r") = ", r"-", r"\sum_{i=0}^{10}", r"1", r"\cdot", r"\log 0",
                       r" = ", r"\infty",
                       font_size=72)
        self.play(TransformMatchingTex(p1q0_, p1q0, transform_mismatches=True), run_time=1.0)
        self.wait(1.0)
        self.play(Unwrite(p1q0), run_time=1.0)
