"""
training1.py
- Gender Classification Model (PyTorch CNN, UTKFace dataset)
- Violence Detection Model (TensorFlow ConvLSTM, Real-Life Violence dataset)
Run this file to train both models and save them.
"""

import os
import glob
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

import kagglehub

# ── Hyper-parameters ─────────────────────────────────────────────────────────
GENDER_EPOCHS   = 10
GENDER_LR       = 0.001
GENDER_BATCH    = 32
VIOLENCE_EPOCHS = 12
VIOLENCE_BATCH  = 4
SEQ_LEN         = 5
FRAME_SIZE      = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Gender Classification (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)
        # UTKFace filename: age_gender_race_date.jpg (gender: 0=Male, 1=Female)
        parts = filename.split("_")
        gender = int(parts[1]) if len(parts) >= 2 else 0
        if self.transform:
            image = self.transform(image)
        return image, gender


class GenderClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


def train_gender_model():
    print("Downloading UTKFace dataset …")
    path = kagglehub.dataset_download("jangedoo/utkface-new")
    data_dir = os.path.join(path, "utkface_aligned_cropped", "UTKFace")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = UTKFaceDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=GENDER_BATCH, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=GENDER_BATCH, shuffle=False, num_workers=2)

    model = GenderClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=GENDER_LR)

    for epoch in range(GENDER_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{GENDER_EPOCHS}]  Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    save_path = "gender_classification.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Gender model saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Violence Detection (TensorFlow ConvLSTM)
# ══════════════════════════════════════════════════════════════════════════════

class VideoDataGenerator(Sequence):
    def __init__(self, dataset_path, classes, batch_size=VIOLENCE_BATCH,
                 sequence_length=SEQ_LEN):
        self.dataset_path    = dataset_path
        self.classes         = classes
        self.batch_size      = batch_size
        self.sequence_length = sequence_length
        self.video_paths     = []
        self.labels          = []
        self._prepare()

    def _prepare(self):
        for label, category in enumerate(self.classes):
            cat_path = os.path.join(self.dataset_path, category)
            for f in os.listdir(cat_path):
                if f.endswith(".mp4"):
                    self.video_paths.append(os.path.join(cat_path, f))
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_vids   = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for vpath, label in zip(batch_vids, batch_labels):
            cap     = cv2.VideoCapture(vpath)
            frames  = []
            count   = 0
            ok, frm = cap.read()
            while ok and len(frames) < self.sequence_length:
                if count % 3 == 0:
                    frames.append(cv2.resize(frm, FRAME_SIZE))
                count += 1
                ok, frm = cap.read()
            cap.release()
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
            X.append(np.array(frames))
            y.append(label)
        return (np.array(X, dtype=np.float32) / 255.0,
                tf.keras.utils.to_categorical(y, num_classes=2))


def build_violence_model():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = Sequential([
        ConvLSTM2D(64, (3, 3), activation="relu", return_sequences=True,
                   input_shape=(SEQ_LEN, *FRAME_SIZE, 3)),
        BatchNormalization(),
        Dropout(0.3),
        ConvLSTM2D(32, (3, 3), activation="relu", return_sequences=False),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax", dtype="float32"),   # keep output in float32
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_violence_model():
    print("Downloading Violence dataset …")
    path = kagglehub.dataset_download(
        "mohamedmustafa/real-life-violence-situations-dataset")
    dataset_path = os.path.join(
        path, "Real Life Violence Dataset")
    classes = ["Violence", "NonViolence"]

    gen   = VideoDataGenerator(dataset_path, classes, batch_size=VIOLENCE_BATCH)
    model = build_violence_model()
    model.summary()
    model.fit(gen, epochs=VIOLENCE_EPOCHS)

    # Evaluate on the same generator (quick sanity check)
    loss, acc = model.evaluate(gen, verbose=0)
    print(f"Training Accuracy: {acc * 100:.2f}%")

    save_path = "violence_detection.keras"   # native Keras format (not .h5 / .pth)
    model.save(save_path)
    print(f"Violence model saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_gender_model()
    train_violence_model()
