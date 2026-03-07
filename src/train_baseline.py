import tensorflow as tf

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "data/facial_emotion_recognition/train",
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale"
)

# Load testing dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "data/facial_emotion_recognition/test",
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale"
)

# Print emotion classes detected from folder names
print("Class names:", train_data.class_names)