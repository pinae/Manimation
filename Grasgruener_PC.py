from manim import Scene
from manim import Text, VGroup, Line
from manim import LEFT, RIGHT, UP, DOWN, BOLD
from manim import Write, Create

class gpuTable(Scene):
    def construct(self):
        self.camera.background_color = '#095748'
        column_labels = ["Grafikkarte", "relative Performance", "Straßenpreis", "Preis/Performance"]
        data = [
            ["4070",    1.38, 550, 398.6],
            ["3080 Ti", 1.65, 550, 333.3],
            ["6950 XT", 1.65, 550, 333.3],
            ["A770",    0.84, 275, 327.4],
            ["7700 XT", 1.22, 380, 311.5],
            ["6750 XT", 1.06, 330, 311.3],
            ["6900 XT", 1.53, 475, 310.5],
            ["3080",    1.47, 450, 306.1],
            ["6700 XT", 1.00, 300, 300.0],
            ["6800 XT", 1.42, 425, 299.3],
            ["3070 Ti", 1.21, 350, 289.3],
            ["3070",    1.13, 325, 287.6],
            ["6800",    1.25, 350, 280.0],
            ["A750",    0.74, 200, 270.3],
            ["7600",    0.89, 230, 258.4],
            ["3060 Ti", 0.97, 250, 257.7],
            ["6650 XT", 0.88, 220, 250.0],
            ["6600 XT", 0.81, 200, 246.9]
        ]
        data = [
            ["6600 XT", 0.81, 200, 246.9],
            ["A750",    0.74, 200, 270.3],
            ["6650 XT", 0.88, 220, 250.0],
            ["7600",    0.89, 230, 258.4],
            ["3060 Ti", 0.97, 250, 257.7],
            ["A770",    0.84, 275, 327.4],
            ["6700 XT", 1.00, 300, 300.0],
            ["3070",    1.13, 325, 287.6],
            ["6750 XT", 1.06, 330, 311.3],
            ["3070 Ti", 1.21, 350, 289.3],
            ["6800",    1.25, 350, 280.0],
            ["7700 XT", 1.22, 380, 311.5],
            ["6800 XT", 1.42, 425, 299.3],
            ["3080",    1.47, 450, 306.1],
            ["6900 XT", 1.53, 475, 310.5],
            ["3080 Ti", 1.65, 550, 333.3],
            ["6950 XT", 1.65, 550, 333.3],
            ["4070",    1.38, 550, 398.6]
        ]
        font_settings = {"font": "Saira ExtraCondensed", "font_size": 28}
        label_line = []
        for i, label in enumerate(column_labels):
            t = Text(label, **font_settings, weight=BOLD, should_center=True).move_to(3.6*UP+5*LEFT+13/4*i*RIGHT)
            label_line.append(t)
        dividers = []
        for i in range(len(data) + 1):
            dividers.append(
                Line(6.5 * LEFT + 3.4 * UP + i * 0.4 * DOWN,
                     6.5 * RIGHT + 3.4 * UP + i * 0.4 * DOWN,
                     stroke_width=.5))
        for i in range(len(column_labels) + 1):
            dividers.append(
                Line(6.5 * LEFT + 3.4 * UP + i * 13/4 * RIGHT,
                     6.5 * LEFT + 3.4 * UP + len(data) * 0.4 * DOWN + i * 13/4 * RIGHT,
                     stroke_width=.5))
        self.play([Write(t) for t in label_line],
                  [Create(d) for d in dividers],
                  run_time=0.5)
        mols = []
        for line_no, line in enumerate(data):
            mol = VGroup()
            for i, d in enumerate(line):
                t = Text(d if type(d) is str else f"{d}", **font_settings, should_center=True).move_to(
                    3.2*UP+line_no*0.4*DOWN+5*LEFT+13/4*i*RIGHT)
                mol.add(t)
            self.play(Write(mol), run_time=0.2)
            mols.append(mol)
        self.wait(1)
        old_positions = [mol.get_center() for mol in mols]
        def data_key(mol : VGroup):
            for i, line in enumerate(data):
                if "".join(line[0].split()) == "".join(mol[0].text.split()):
                    return line[3]
            print(mol[0].text)
            return 0.0
        mols.sort(key=data_key)
        self.play([mol.animate.move_to([mol.get_x(), old_positions[i][1], mol.get_z()]) for i, mol in enumerate(mols)],
                  run_time=1)
        self.wait(5)
