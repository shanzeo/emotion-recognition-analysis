import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "data/facial_emotion_recognition/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(48, 48),
    batch_size=8,
    color_mode="rgb"
)

# Load validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    "data/facial_emotion_recognition/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(48, 48),
    batch_size=8,
    color_mode="rgb"
)

print("Class names:", train_data.class_names)

# Data augmentation (helps with small datasets)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Build CNN model
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(48, 48, 3)),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.Dense(8, activation="softmax")  # 8 emotion classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Plot accuracy
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()