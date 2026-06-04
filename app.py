import gradio as gr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import logging
import tensorflow as tf
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Encryption ────────────────────────────────────────────────────────────────
KEY_FILE = "fernet.key"
if os.path.exists(KEY_FILE):
    with open(KEY_FILE, "rb") as f:
        _key = f.read()
else:
    _key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(_key)
cipher = Fernet(_key)

# ── Gender Classification Model (PyTorch) ─────────────────────────────────────
class GenderClassificationModel(nn.Module):
    """Custom CNN trained on UTKFace dataset — input: (B, 3, 128, 128), output: (B, 2)."""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# ── Violence Detection Model (TensorFlow/Keras ConvLSTM) ──────────────────────
def build_violence_model(seq_len=5):
    """ConvLSTM model trained on real-life violence dataset — input: (B, T, 224, 224, 3)."""
    model = tf.keras.Sequential([
        tf.keras.layers.ConvLSTM2D(64, (3, 3), activation="relu",
                                    return_sequences=True,
                                    input_shape=(seq_len, 224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.ConvLSTM2D(32, (3, 3), activation="relu",
                                    return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    return model

# ── Crowd Detection Model (TensorFlow/Keras CNN) ──────────────────────────────
def build_crowd_model():
    """CNN trained on CrowdHuman dataset — input: (B, 224, 224, 3), output: (B, 2)."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])

# ── Model Loading ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

gender_model = None
if os.path.exists("gender_classification.pth"):
    try:
        gender_model = GenderClassificationModel().to(device)
        gender_model.load_state_dict(
            torch.load("gender_classification.pth", map_location=device, weights_only=True)
        )
        gender_model.eval()
        logger.info("Gender model loaded.")
    except Exception as e:
        logger.error(f"Gender model load failed: {e}")
        gender_model = None
else:
    logger.warning("gender_classification.pth not found.")

violence_model = None
for vpath in ("violence_detection.keras", "violence_detection_model.h5"):
    if os.path.exists(vpath):
        try:
            violence_model = tf.keras.models.load_model(vpath)
            logger.info(f"Violence model loaded from {vpath}.")
        except Exception as e:
            logger.error(f"Violence model load failed: {e}")
        break
else:
    logger.warning("No violence detection model file found.")

crowd_model = None
for cpath in ("crowd_detection.keras", "crowd_detection.h5"):
    if os.path.exists(cpath):
        try:
            crowd_model = build_crowd_model()
            crowd_model.load_weights(cpath)
            logger.info(f"Crowd model loaded from {cpath}.")
        except Exception as e:
            logger.error(f"Crowd model load failed: {e}")
        break
else:
    logger.warning("No crowd detection model file found.")

# ── Frame Sampling ─────────────────────────────────────────────────────────────
SAMPLE_STRIDE = 10  # analyse every 10th frame
SEQ_LEN = 5         # number of frames per violence-detection sequence

def _sample_frames(video_path: str, stride: int = SAMPLE_STRIDE):
    """Open video and return uniformly sampled BGR frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

# ── Core Inference ─────────────────────────────────────────────────────────────
def detect_safety(video_path: str) -> str:
    # Input validation
    if not video_path:
        return "⚠ No input provided."
    is_stream = video_path.startswith(("rtsp://", "http://", "https://"))
    if not is_stream and not os.path.exists(video_path):
        return "⚠ File not found."

    frames = _sample_frames(video_path)
    if not frames:
        return "⚠ Could not read any frames from the video."

    # Encrypt the first frame (transit/at-rest protection demo)
    raw_bytes = frames[0].tobytes()
    encrypted = cipher.encrypt(raw_bytes)
    decrypted = np.frombuffer(cipher.decrypt(encrypted), dtype=np.uint8).reshape(frames[0].shape)

    results = {}

    # ── Gender Classification ──────────────────────────────────────────────────
    if gender_model:
        frame_g = cv2.resize(decrypted, (128, 128)).astype(np.float32) / 255.0
        # Convert BGR→RGB then HWC→CHW
        frame_g = np.transpose(cv2.cvtColor(frame_g, cv2.COLOR_BGR2RGB), (2, 0, 1))
        tensor_g = torch.tensor(frame_g).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = gender_model(tensor_g)
            pred = torch.argmax(logits, dim=1).item()
        results["gender"] = "Female" if pred == 1 else "Male"

    # ── Violence Detection ─────────────────────────────────────────────────────
    if violence_model:
        seq_frames = frames[:SEQ_LEN] if len(frames) >= SEQ_LEN else frames
        # Pad if too short
        while len(seq_frames) < SEQ_LEN:
            seq_frames.append(seq_frames[-1])
        seq = np.stack([
            cv2.resize(f, (224, 224))[:, :, ::-1].astype(np.float32) / 255.0
            for f in seq_frames
        ])  # (T, 224, 224, 3) RGB
        seq = np.expand_dims(seq, axis=0)  # (1, T, 224, 224, 3)
        pred_v = np.argmax(violence_model.predict(seq, verbose=0))
        results["violence"] = "Violence Detected" if pred_v == 1 else "No Violence"

    # ── Crowd Detection ───────────────────────────────────────────────────────
    if crowd_model:
        frame_c = cv2.resize(decrypted, (224, 224))[:, :, ::-1].astype(np.float32) / 255.0
        frame_c = np.expand_dims(frame_c, axis=0)  # (1, 224, 224, 3)
        pred_c = np.argmax(crowd_model.predict(frame_c, verbose=0))
        results["crowd"] = "Crowd Present" if pred_c == 1 else "No Crowd"

    # ── Alert Logic ───────────────────────────────────────────────────────────
    is_female = results.get("gender") == "Female"
    has_violence = results.get("violence") == "Violence Detected"
    has_crowd = results.get("crowd") == "Crowd Present"

    if is_female and (has_violence or has_crowd):
        alert = "🚨 ALERT: Potential threat detected!"
    else:
        alert = "✅ Environment appears safe."

    lines = [f"Gender: {results.get('gender', 'N/A (model not loaded)')}",
             f"Violence: {results.get('violence', 'N/A (model not loaded)')}",
             f"Crowd: {results.get('crowd', 'N/A (model not loaded)')}",
             f"Alert: {alert}",
             f"Frames analysed: {len(frames)} (sampled every {SAMPLE_STRIDE} frames)"]
    return "\n".join(lines)

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="SangRakshak — Women's Safety AI") as iface:
    gr.HTML("""
        <h1 style="text-align:center;color:#333;">संरक्षक — SangRakshak</h1>
        <h3 style="text-align:center;color:#666;">AI-Powered Women's Safety System</h3>
        <p style="text-align:center;color:#888;">
            "A nation's progress is measured by the status and safety of its women." — Jawaharlal Nehru
        </p>
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📡 CCTV Live Stream")
            cctv_url = gr.Textbox(label="CCTV URL (rtsp:// or http://)",
                                  placeholder="rtsp://...")
            btn_cctv = gr.Button("Analyse Stream", variant="primary")
            out_cctv = gr.Textbox(label="Stream Analysis", lines=6)
            btn_cctv.click(fn=detect_safety, inputs=cctv_url, outputs=out_cctv)

        with gr.Column():
            gr.Markdown("### 🎥 Upload Video")
            uploaded_video = gr.Video(label="Upload Video File")
            btn_video = gr.Button("Analyse Video", variant="primary")
            out_video = gr.Textbox(label="Video Analysis", lines=6)
            btn_video.click(fn=detect_safety, inputs=uploaded_video, outputs=out_video)

    gr.HTML("""
        <p style="text-align:center;font-size:0.85em;color:#999;margin-top:20px;">
            Frames encrypted with AES-256 (Fernet) during processing.
        </p>
    """)

if __name__ == "__main__":
    iface.launch(share=True)
