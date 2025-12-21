import tensorflow as tf
import numpy as np
import os
import csv
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = "model/animal_model_2.keras"
TEST_DIR = "data/test"
OUTPUT_CSV = "test_results_model_2.csv"
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "squirrel"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Lists for metric calculation
y_true = []
y_pred = []

# Create csv file
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter=";")

    # Csv headers
    writer.writerow([
        "image_path",
        "true_label",
        "predicted_label",
        "confidence"
    ])

    # Iterate all classes in test directory
    for true_label in os.listdir(TEST_DIR):
        class_dir = os.path.join(TEST_DIR, true_label)

        if not os.path.isdir(class_dir):
            continue

        for image_name in os.listdir(class_dir):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(class_dir, image_name)

            # Load image
            img = load_img(image_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)

            # Normalize image
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions)
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = predictions[0][predicted_index]

            # Store values for metrics
            y_true.append(CLASS_NAMES.index(true_label))
            y_pred.append(predicted_index)

            # Write prediction to csv
            writer.writerow([
                image_path,
                true_label,
                predicted_label,
                f"{confidence:.4f}"
            ])

# Calculate metrics using test set
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

# Print metrics
print("Model Performance on Test Dataset")
print(f"accuracy  : {accuracy:.4f}")
print(f"precision : {precision:.4f}")
print(f"recall    : {recall:.4f}")
print(f"f1 score  : {f1:.4f}")

print(f"Test completed. Results saved to '{OUTPUT_CSV}'")
