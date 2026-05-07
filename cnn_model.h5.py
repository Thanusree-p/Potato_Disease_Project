# ============================================
# POTATO LEAF DISEASE DETECTION USING CNN
# ============================================

# =========================
# IMPORT LIBRARIES
# =========================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# =========================
# DATASET PATH
# =========================

import os

dataset_path = os.path.join("dataset", "PotatoPlants")


# =========================
# IMAGE DATA GENERATOR
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)


# =========================
# TRAIN GENERATOR
# =========================

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


# =========================
# VALIDATION GENERATOR
# =========================

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# =========================
# PRINT CLASS INDICES
# =========================

print(train_generator.class_indices)


# =========================
# CNN MODEL
# =========================

cnn_model = Sequential()

# First Convolution Layer
cnn_model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(224,224,3)
    )
)

cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2,2)))


# Second Convolution Layer
cnn_model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2,2)))


# Third Convolution Layer
cnn_model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2,2)))


# Fourth Convolution Layer
cnn_model.add(
    Conv2D(
        256,
        (3,3),
        activation='relu'
    )
)

cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2,2)))


# Flatten Layer
cnn_model.add(Flatten())


# Dense Layer
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(0.5))


# Dense Layer
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.3))


# Output Layer
cnn_model.add(Dense(3, activation='softmax'))


# =========================
# COMPILE MODEL
# =========================

cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# =========================
# MODEL SUMMARY
# =========================

cnn_model.summary()


# =========================
# TRAIN MODEL
# =========================

history = cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)


# =========================
# SAVE MODEL
# =========================

cnn_model.save("cnn_model.h5")

print("Model Saved Successfully")


# =========================
# PLOT ACCURACY
# =========================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()

plt.show()


# =========================
# PLOT LOSS
# =========================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.show()