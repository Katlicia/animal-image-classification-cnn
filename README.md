# Animal Image Classification with CNN

This project is an image classification system that uses a Convolutional Neural Network (CNN) to classify different animal species from images.
The model is implemented using TensorFlow and Keras.

---

## Project Overview

The goal of this project is to train a CNN model capable of recognizing multiple animal classes from image data.
The project includes dataset preparation, model training, evaluation using multiple metrics, and prediction on test images.

---

## Dataset Structure

The dataset is organized using a directory-based structure compatible with TensorFlow.

Each class folder contains images belonging to that class.

---

## Model Architecture

The model is a custom Convolutional Neural Network consisting of:
- Input layer with image shape (128, 128, 3)
- Three convolutional blocks (Conv2D + ReLU + MaxPooling)
- Fully connected layers with Dropout
- Softmax output layer for multi-class classification

---

## Training Configuration

- Image size: 128 x 128
- Batch size: 32
- Optimizer: Adam
- Loss function: sparse_categorical_crossentropy
- Metrics: accuracy
- Epochs: 30

Input images are normalized to the range [0, 1].

---

## Evaluation Metrics

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

---

## Prediction

Test images are processed automatically and prediction results are saved to a CSV file using ';' as delimiter.

CSV format:
image_path ; true_label ; predicted_label ; confidence

---

## How to Run

### Setup a virtual environment in the project folder
```
python -m venv venv
```

### Activate the virtual environment

### Windows
```
venv\Scripts\activate
```
### macOS / Linux
```
source venv/bin/activate
```

### Download the requirements
After setting up your venv download the libraries
```
pip install -r requirements.txt
``` 

### Prepare dataset inside the data directory
1. Run the install-dataset.py to download the dataset.
2. Cut the folder before the class folders.
3. Print it in the project folder.
4. Rename the Italian class names to English.
5. Run the create-test-train-datas.py to seperate the train and test datas.
6. The code creates 80% train data to 20% test data.

### Train the model (Or use the pre-trained model)
```
python model/train.py
```
### Run test datas:
```
python model/detect.py
```
### Create a web interface with app.py
```
python app.py
```

## Requirements

- Python 3.x
- KaggleHub (To download the dataset)
- TensorFlow
- NumPy
- Scikit-learn
- Streamlit
- Pillow