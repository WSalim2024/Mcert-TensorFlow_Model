import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from cifar_model import Net
import os

# 1. Load Model
PATH = './cifar_net.pth'
if not os.path.exists(PATH):
    raise FileNotFoundError("Run pytorch_train.py first!")

# Initialize the architecture and load weights
net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()  # Set to evaluation mode

# CIFAR-10 Classes
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CifarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch Object Sketcher (CIFAR-10)")
        self.root.geometry("600x700")

        tk.Label(root, text="Draw an Object (Car, Ship, Bird...)", font=("Helvetica", 20, "bold")).pack(pady=10)

        # Canvas
        self.canvas = tk.Canvas(root, width=500, height=500, bg='black', cursor="cross")
        self.canvas.pack()

        self.image1 = Image.new("RGB", (500, 500), "black")  # RGB Mode for CIFAR
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.label_result = tk.Label(root, text="Prediction: ?", font=("Helvetica", 14), fg="blue")
        self.label_result.pack(pady=15)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="PREDICT", command=self.predict, bg='green', fg='white').pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="CLEAR", command=self.clear, width=10).pack(side=tk.LEFT, padx=10)

    def paint(self, event):
        r = 12
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def clear(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (500, 500), "black")
        self.draw = ImageDraw.Draw(self.image1)
        self.label_result.config(text="Prediction: ?")

    def predict(self):
        # 1. Resize to 32x32 (CIFAR standard)
        img = self.image1.resize((32, 32))

        # 2. Transform to Tensor and Normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 32, 32)

        # 3. Predict
        with torch.no_grad():
            outputs = net(img_tensor)
            _, predicted = torch.max(outputs, 1)

        result = CLASSES[predicted.item()]
        self.label_result.config(text=f"I think this is a: {result.upper()}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CifarApp(root)
    root.mainloop()