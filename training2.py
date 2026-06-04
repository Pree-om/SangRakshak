"""
training2.py
- Crowd Detection Model (TensorFlow CNN, CrowdHuman dataset)
Run this file to train the crowd detection model and save it.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

import kagglehub

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4


def load_images(folder: str, label: int, limit: int = 5000):
    """Load images from folder, resize, return (arrays, labels)."""
    X, y = [], []
    exts = (".jpg", ".jpeg", ".png")
    for fname in os.listdir(folder):
        if not fname.lower().endswith(exts):
            continue
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
        X.append(img)
        y.append(label)
        if len(X) >= limit:
            break
    return X, y


def build_crowd_model():
    return Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(2, activation="softmax"),
    ])


def train_crowd_model():
    print("Downloading CrowdHuman dataset …")
    path = kagglehub.dataset_download("permanalwep/crowdhuman-crowd-detection")
    data_root = os.path.join(
        path, "versions", "5",
        "CrowdHuman Cropped", "Dataset CrowdHuman")

    crowd_dir     = os.path.join(data_root, "crowd")
    non_crowd_dir = os.path.join(data_root, "non crowd")

    print("Loading crowd images …")
    X_crowd,  y_crowd  = load_images(crowd_dir,     label=1)
    print("Loading non-crowd images …")
    X_none,   y_none   = load_images(non_crowd_dir, label=0)

    X = np.array(X_crowd + X_none, dtype=np.float32) / 255.0
    y = np.array(y_crowd + y_none, dtype=np.int32)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_crowd_model()
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]
    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val),
              callbacks=callbacks)

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    save_path = "crowd_detection.keras"
    model.save(save_path)
    print(f"Crowd model saved → {save_path}")


if __name__ == "__main__":
    train_crowd_model()
