import tensorflow as tf
import os
from tensorflow.keras import layers # type: ignore

from cnn_model import create_cnn_model

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Make sure train and test folders are under the data folder (data/test data/train)
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1)
# ])

# Normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)

# train_ds = train_ds.map(lambda x, y: (normalization_layer((data_augmentation(x, training=True))), y))

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = create_cnn_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 30

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

model.save("animal_model_2.keras")
