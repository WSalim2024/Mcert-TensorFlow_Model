import tensorflow as tf
import numpy as np
from emnist import extract_training_samples, extract_test_samples

# 1. Load EMNIST 'balanced' (47 classes: 0-9, A-Z, and some a-z)
print("Loading EMNIST data... this might take a moment.")
x_train, y_train = extract_training_samples('balanced')
x_test, y_test = extract_test_samples('balanced')

# 2. Preprocess
# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# CRITICAL FIX: EMNIST images are rotated and flipped by default.
# We must transpose them to match how you draw in the GUI.
x_train = np.array([np.transpose(img) for img in x_train])
x_test = np.array([np.transpose(img) for img in x_test])

# 3. Build Model (Slightly larger to handle the complexity of letters)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'), # Increased neurons
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax') # 47 Classes now
])

# 4. Compile
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# 5. Train
print("Starting training on Characters and Digits...")
model.fit(x_train, y_train, epochs=20) # Increased epochs for better learning

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# 7. Save
model.save('my_emnist_model.h5') # New filename
print("Model saved as my_emnist_model.h5")