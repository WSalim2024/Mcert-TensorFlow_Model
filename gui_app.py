import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import os

# 1. Load the NEW EMNIST model
MODEL_PATH = 'my_emnist_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please run the new main.py first!")

model = tf.keras.models.load_model(MODEL_PATH)

# 2. The Decoder Map (EMNIST Balanced Mapping)
# 0-9 are digits, 10-35 are A-Z, 36-46 are a-z (only the ones that look different)
EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Character Recognizer (OCR)")
        self.root.geometry("600x700")
        self.root.configure(bg='#f0f0f0')

        # Title
        tk.Label(root, text="Draw a Character (0-9, A-Z)", font=("Helvetica", 20, "bold"), bg='#f0f0f0').pack(pady=10)

        # Canvas
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height,
                                bg='black', cursor="cross", relief="sunken", borderwidth=2)
        self.canvas.pack(pady=5)

        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind("<B1-Motion>", self.paint)

        # Result Label (Small font as requested)
        self.label_result = tk.Label(root, text="Prediction: ?",
                                     font=("Helvetica", 12, "bold"), fg="blue", bg='#f0f0f0')
        self.label_result.pack(pady=15)

        # Buttons
        control_frame = tk.Frame(root, bg='#f0f0f0')
        control_frame.pack(pady=5)

        self.predict_btn = tk.Button(control_frame, text="PREDICT", command=self.predict_char,
                                     bg='green', fg='white', font=("Helvetica", 14), width=15)
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        self.clear_btn = tk.Button(control_frame, text="CLEAR", command=self.clear_canvas,
                                   font=("Helvetica", 14), width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=10)

    def paint(self, event):
        brush_size = 12
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image1)
        self.label_result.config(text="Prediction: ?")

    def predict_char(self):
        # Resize and Preprocess
        img = self.image1.resize((28, 28))
        img_array = np.array(img) / 255.0

        # TRANSPOSE IS KEY: The model was trained on transposed images, so we must transpose the input too
        img_array = np.transpose(img_array)

        img_array = img_array.reshape(1, 28, 28)

        # Predict
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Decode the result
        predicted_char = EMNIST_MAPPING.get(result_index, "?")

        self.label_result.config(text=f"I see: {predicted_char}\n({confidence:.1f}% sure)")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()