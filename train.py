import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils.feature_extraction import extract_features

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/RAVDESS/"
MODEL_SAVE_PATH = "models/emotion_model.h5"
NUM_CLASSES = 6

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
}

# ─── Load data ─────────────────────────────────────────────────────────────────
print("📂 Loading dataset…")
features, labels = [], []

for actor_folder in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for fname in os.listdir(actor_path):
        if not fname.endswith(".wav"):
            continue
        parts = fname.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in EMOTIONS:
            continue
        fpath = os.path.join(actor_path, fname)
        feat = extract_features(fpath)
        if feat is not None:
            features.append(feat)
            labels.append(int(emotion_code) - 1)

X = np.array(features)               # shape: (N, 180)
y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)

print(f"✅ Loaded {len(X)} samples | Feature shape: {X.shape}")

# ─── Reshape for LSTM: (N, timesteps=180, features=1) ──────────────────────────
X = X.reshape(X.shape[0], X.shape[1], 1)

# ─── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
)

# ─── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)
model.summary()

# ─── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
]

# ─── Train ─────────────────────────────────────────────────────────────────────
print("🚀 Training…")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# ─── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")
print(f"✅ Best model saved to: {MODEL_SAVE_PATH}")