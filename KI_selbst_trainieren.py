from manim import Code, Text, Tex
from manim import MoveToTarget, FadeOut, Create, Transform
from manim import UP, DOWN, LEFT, RIGHT
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
        self.camera.background_color = '#404040'
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
        full_code.target.shift(14 * UP)
        self.play(MoveToTarget(full_code), run_time=2)
        self.play(FadeOut(full_code), run_time=0.5)
        self.wait(duration=1)

        code = '''import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor'''
        c1 = make_code(code)
        self.play(Create(c1), run_time=2)
        self.wait(duration=2)

        code = '''training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)'''
        c2 = make_code(code, 7)
        #c2.next_to(c1, direction=DOWN)
        c1.generate_target()
        c1.target.next_to(c2, direction=UP)
        self.play(MoveToTarget(c1), run_time=1)
        self.play(Create(c2), run_time=2)
        self.wait(duration=2)
        c2.code += '''
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)'''
        self.play(c2.animate, run_time=2)
        self.wait(duration=2)
