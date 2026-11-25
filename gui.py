"""Tkinter canvas for inference with the saved MNIST model."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw
import torch

from model import MNISTClassifier


MODEL_PATH = Path(__file__).resolve().parent / "model" / "mnist_model.pth"
CANVAS_SIZE = 280
BRUSH_SIZE = 18


def prepare_image(image):
    """Fit the digit into 20x20 pixels and center its mass on a 28x28 image."""
    image = image.convert("L")
    bounds = image.getbbox()
    if bounds is None:
        raise ValueError("Najprv nakresli číslicu.")
    digit = image.crop(bounds)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    centered = Image.new("L", (28, 28))
    centered.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    pixels = torch.tensor(list(centered.getdata()), dtype=torch.float32).reshape(28, 28)
    mass = pixels.sum()
    coordinates = torch.arange(28, dtype=torch.float32)
    center_x = (pixels.sum(dim=0) * coordinates).sum() / mass
    center_y = (pixels.sum(dim=1) * coordinates).sum() / mass
    shifted = Image.new("L", (28, 28))
    shifted.paste(centered, (round(13.5 - center_x.item()), round(13.5 - center_y.item())))
    tensor = torch.tensor(list(shifted.getdata()), dtype=torch.float32).reshape(1, 1, 28, 28)
    return (tensor / 255.0 - 0.1307) / 0.3081, shifted


def load_model(path=MODEL_PATH):
    model = MNISTClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


class MNISTApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.last_point = None
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE))
        self.draw = ImageDraw.Draw(self.image)
        root.title("MNIST – rozpoznávanie číslic")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=20)
        frame.grid()
        ttk.Label(frame, text="Nakresli číslicu 0–9", font=("Arial", 18, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 8))
        ttk.Label(frame, text="Kresli ľavým tlačidlom myši. Enter = rozpoznať, Esc = vymazať.").grid(
            row=1, column=0, columnspan=2, pady=(0, 14))
        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                background="black", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=2, column=0, padx=(0, 20))
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        root.bind("<ButtonRelease-1>", self.end_stroke)
        panel = ttk.Frame(frame)
        panel.grid(row=2, column=1, sticky="n")
        self.result = tk.StringVar(value="Výsledok: —")
        ttk.Label(panel, textvariable=self.result, font=("Arial", 16, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 10))
        self.scores = []
        for digit in range(10):
            ttk.Label(panel, text=str(digit)).grid(row=digit + 1, column=0, padx=5)
            bar = ttk.Progressbar(panel, length=130, maximum=100)
            bar.grid(row=digit + 1, column=1)
            label = ttk.Label(panel, text="—", width=7, anchor="e")
            label.grid(row=digit + 1, column=2)
            self.scores.append((bar, label))
        ttk.Button(frame, text="Rozpoznať", command=self.predict).grid(row=3, column=0, sticky="ew", pady=12)
        ttk.Button(frame, text="Vymazať", command=self.clear).grid(row=3, column=1, sticky="ew", pady=12)
        self.preview = ttk.Label(frame)
        self.preview.grid(row=4, column=0)
        ttk.Label(frame, text="Náhľad vstupu modelu (28 × 28).\nPercentá sú skóre modelu, nie záruka správnosti.").grid(
            row=4, column=1, sticky="w")
        root.bind("<Return>", lambda event: self.predict())
        root.bind("<Escape>", lambda event: self.clear())

    def reset_result(self):
        self.result.set("Výsledok: —")
        for bar, label in self.scores:
            bar["value"] = 0
            label.configure(text="—")
        self.preview.configure(image="")
        self.preview.image = None

    def start_stroke(self, event):
        self.last_point = None
        self.paint(event)

    def paint(self, event):
        x = max(0, min(CANVAS_SIZE - 1, event.x))
        y = max(0, min(CANVAS_SIZE - 1, event.y))
        radius = BRUSH_SIZE / 2
        if self.last_point is not None:
            self.canvas.create_line(*self.last_point, x, y, fill="white", width=BRUSH_SIZE)
            self.draw.line([self.last_point, (x, y)], fill=255, width=BRUSH_SIZE)
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*bounds, fill="white", outline="white")
        self.draw.ellipse(bounds, fill=255)
        self.last_point = (x, y)
        self.reset_result()

    def end_stroke(self, event):
        self.last_point = None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE))
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.reset_result()

    def predict(self):
        try:
            tensor, preview = prepare_image(self.image)
        except ValueError as error:
            self.result.set(str(error))
            return
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        self.result.set(f"Výsledok: {probabilities.argmax().item()}")
        for probability, (bar, label) in zip(probabilities.tolist(), self.scores):
            bar["value"] = probability * 100
            label.configure(text=f"{probability:.1%}")
        # Tk reads PPM directly, including systems without Pillow's ImageTk module.
        thumbnail = preview.resize((84, 84), Image.Resampling.NEAREST).convert("RGB")
        ppm = b"P6\n84 84\n255\n" + thumbnail.tobytes()
        photo = tk.PhotoImage(master=self.root, data=ppm, format="PPM")
        self.preview.configure(image=photo)
        self.preview.image = photo


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        model = load_model()
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        messagebox.showerror("Model sa nepodarilo načítať",
                             f"Skontroluj váhy modelu:\n{MODEL_PATH}\n\n{error}", parent=root)
        root.destroy()
        return
    MNISTApp(root, model)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()