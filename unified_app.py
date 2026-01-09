import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import numpy as np
import os

# --- AI IMPORTS ---
# We wrap these in try-blocks so the app opens even if one framework is broken
print("Initializing AI Engines...")

# 1. TensorFlow Setup (Characters)
try:
    import tensorflow as tf

    TF_MODEL_PATH = 'my_emnist_model.h5'
    if os.path.exists(TF_MODEL_PATH):
        tf_model = tf.keras.models.load_model(TF_MODEL_PATH)
        print("✅ TensorFlow (EMNIST) loaded.")
    else:
        tf_model = None
        print("⚠️ TensorFlow model file missing.")
except ImportError:
    tf_model = None
    print("❌ TensorFlow library not found.")

# 2. PyTorch Setup (Objects)
try:
    import torch
    import torchvision.transforms as transforms
    from cifar_model import Net  # Import the class we created earlier

    TORCH_MODEL_PATH = './cifar_net.pth'
    if os.path.exists(TORCH_MODEL_PATH):
        torch_model = Net()
        torch_model.load_state_dict(torch.load(TORCH_MODEL_PATH))
        torch_model.eval()  # Set to evaluation mode
        print("✅ PyTorch (CIFAR-10) loaded.")
    else:
        torch_model = None
        print("⚠️ PyTorch model file missing.")
except ImportError:
    torch_model = None
    print("❌ PyTorch library not found.")

# --- MAPPINGS ---
EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class UnifiedAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Dashboard: Hybrid Recognition System")
        self.root.geometry("600x750")
        self.root.configure(bg='#e6e6e6')

        # --- HEADER ---
        header_frame = tk.Frame(root, bg='#333', height=80)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="AI BRAIN SELECTOR", font=("Segoe UI", 18, "bold"), fg='white', bg='#333').pack(
            pady=10)

        # --- MODE SWITCHER ---
        self.mode_var = tk.StringVar(value="EMNIST")

        switch_frame = tk.Frame(root, bg='#e6e6e6')
        switch_frame.pack(pady=10)

        # Radio Buttons to toggle engines
        tk.Radiobutton(switch_frame, text="Read Characters (TensorFlow)", variable=self.mode_var,
                       value="EMNIST", command=self.update_ui_mode, font=("Segoe UI", 12), bg='#e6e6e6').pack(
            side=tk.LEFT, padx=10)
        tk.Radiobutton(switch_frame, text="See Objects (PyTorch)", variable=self.mode_var,
                       value="CIFAR", command=self.update_ui_mode, font=("Segoe UI", 12), bg='#e6e6e6').pack(
            side=tk.LEFT, padx=10)

        # --- CANVAS ---
        self.canvas_frame = tk.Frame(root, bg='white', bd=5, relief="groove")
        self.canvas_frame.pack(pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, width=500, height=500, bg='black', cursor="cross")
        self.canvas.pack()

        # We start with a grayscale image backend
        self.image1 = Image.new("RGB", (500, 500), "black")
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

        # --- PREDICTION DISPLAY ---
        self.result_label = tk.Label(root, text="Draw something...", font=("Segoe UI", 16, "bold"), fg="#333",
                                     bg='#e6e6e6')
        self.result_label.pack(pady=10)

        # --- CONTROLS ---
        btn_frame = tk.Frame(root, bg='#e6e6e6')
        btn_frame.pack(pady=10)

        self.predict_btn = tk.Button(btn_frame, text="ACTIVATE AI", command=self.run_prediction,
                                     bg='#007acc', fg='white', font=("Segoe UI", 12, "bold"), width=15)
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="CLEAR", command=self.clear_canvas,
                  bg='#cc0000', fg='white', font=("Segoe UI", 12, "bold"), width=10).pack(side=tk.LEFT, padx=10)

        self.update_ui_mode()  # Set initial state

    def update_ui_mode(self):
        # Update text/colors based on selected model
        mode = self.mode_var.get()
        if mode == "EMNIST":
            self.result_label.config(text="Mode: Character Reader (Draw 0-9, A-Z)")
            self.canvas.config(bg='black')  # White on black is best for digits
        else:
            self.result_label.config(text="Mode: Object Detector (Draw a simple shape)")
            # Note: CIFAR handles black backgrounds okay, but training data was photos.

    def paint(self, event):
        r = 12
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)

        # Draw on UI
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        # Draw on PIL Memory (Always RGB to support both models)
        self.draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255), outline=(255, 255, 255))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (500, 500), "black")
        self.draw = ImageDraw.Draw(self.image1)
        self.result_label.config(text="Canvas Cleared.")

    def run_prediction(self):
        mode = self.mode_var.get()

        if mode == "EMNIST":
            self.predict_tensorflow()
        elif mode == "CIFAR":
            self.predict_pytorch()

    def predict_tensorflow(self):
        if tf_model is None:
            self.result_label.config(text="Error: TensorFlow model not loaded.")
            return

        # 1. Preprocess: Convert RGB to Grayscale ("L") -> Resize -> Normalize
        gray_img = self.image1.convert("L").resize((28, 28))
        img_array = np.array(gray_img) / 255.0

        # 2. Transpose (Crucial for EMNIST)
        img_array = np.transpose(img_array)
        img_array = img_array.reshape(1, 28, 28)

        # 3. Predict
        pred = tf_model.predict(img_array)
        idx = np.argmax(pred)
        conf = np.max(pred) * 100

        char = EMNIST_MAPPING.get(idx, "?")
        self.result_label.config(text=f"TensorFlow sees: {char} ({conf:.1f}%)")

    def predict_pytorch(self):
        if torch_model is None:
            self.result_label.config(text="Error: PyTorch model not loaded.")
            return

        # 1. Preprocess: Resize -> Transform to Tensor
        # Note: We keep it RGB because CIFAR expects 3 channels
        img_resized = self.image1.resize((32, 32))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img_tensor = transform(img_resized).unsqueeze(0)  # Batch dim

        # 2. Predict
        with torch.no_grad():
            outputs = torch_model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        cls = CIFAR_CLASSES[predicted.item()]
        self.result_label.config(text=f"PyTorch sees: {cls.upper()}")


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedAIApp(root)
    root.mainloop()