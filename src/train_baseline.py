import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt

print("loading datasets")
data_folder= "data/facial_emotion_recognition/train"

train_ds =tf.keras.utils.image_dataset_from_directory(
    data_folder,
    validation_split =.2,
    subset="training" ,
    seed= 123,
    image_size=(48 ,48) ,
    batch_size = 8,
    color_mode="rgb")

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_folder ,
    validation_split =.2,
    subset="validation",
    seed =123,
    image_size=( 48,48 ),
    batch_size= 8 ,
    color_mode="rgb")

print("detected emotion classes")
print(train_ds.class_names)

augment_layer= tf.keras.Sequential([
    layers.RandomFlip("horizontal" ) ,
    layers.RandomRotation( .1),
    layers.RandomZoom(.1 )] )
print("building CNN model")

cnn_model= models.Sequential( )

cnn_model.add( augment_layer)

cnn_model.add(layers.Rescaling(1./255 , input_shape=(48 ,48,3) ))
cnn_model.add(layers.Conv2D( 32 , (3, 3), activation="relu" ))
cnn_model.add(layers.MaxPooling2D( ))

cnn_model.add(layers.Conv2D(64 , ( 3,3), activation="relu" ))
cnn_model.add(layers.MaxPooling2D())

cnn_model.add(layers.Flatten() )

cnn_model.add(layers.Dense(64 , activation="relu"))
cnn_model.add(layers.Dense(8, activation="softmax") )  # 8 emotions
cnn_model.compile(
    optimizer="adam" ,
    loss= "sparse_categorical_crossentropy",
    metrics=[ "accuracy" ] )
cnn_model.summary()

print("starting training")

train_history= cnn_model.fit(
    train_ds ,
    validation_data=val_ds ,
    epochs=10 )

plt.figure( )
plt.plot(train_history.history["accuracy"] , label="train accuracy" )
plt.plot(train_history.history["val_accuracy"] , label="val accuracy ")
plt.xlabel("Epoch ")
plt.ylabel(" Accuracy")
plt.title("Training vs validation Accuracy")
plt.legend()
plt.show( )

plt.figure()
plt.plot(train_history.history["loss" ], label="train loss" )
plt.plot(train_history.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title( "Training vs Validation Loss")
plt.legend()
plt.show()