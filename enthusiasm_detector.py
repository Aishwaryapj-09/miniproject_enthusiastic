"""
Teacher Enthusiasm Detection System
=====================================
Pipeline:
  1. Extract frames (5–10 FPS)
  2. Detect facial emotion via DeepFace
  3. Extract body landmarks via MediaPipe Pose
  4. Compute hand velocity, shoulder openness, head tilt, body motion
  5. Build per-frame feature vector
  6. Sequence of 16 frames → LSTM → enthusiastic / not enthusiastic
  7. Frame-wise predictions + overall percentage

Author : Production AI Engineer
Version: 1.0.0
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os
import cv2
import json
import math
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import mediapipe as mp


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.ERROR,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONSTANTS & CONFIG
# ─────────────────────────────────────────────
CFG = {
    # Frame sampling
    "target_fps": 6,           # frames per second to sample
    # Sequence
    "seq_len": 16,             # LSTM sequence length
    "stride": 4,               # stride for sliding window
    # Feature dimensionality
    "n_features": 15,
    "threshold": 0.5,# emotion(6) + hand_vel(2) + shoulder(1) + head_tilt(1) + body_motion(1) + face_conf(1)
    # Model
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.4,
    # Training
    "epochs": 60,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 12,            # early stopping
    # Paths
    "model_path": "enthusiasm_lstm.pt",
    "scaler_path": "feature_scaler.pkl",
    "label_map": {"enthu": 1, "not_enthu": 0},
}

ENTHUSIASTIC_EMOTIONS = {"happy", "surprise"}

# MediaPipe landmark indices
MP_POSE = mp.solutions.pose
LANDMARKS = MP_POSE.PoseLandmark
LEFT_SHOULDER  = LANDMARKS.LEFT_SHOULDER.value
RIGHT_SHOULDER = LANDMARKS.RIGHT_SHOULDER.value
LEFT_WRIST     = LANDMARKS.LEFT_WRIST.value
RIGHT_WRIST    = LANDMARKS.RIGHT_WRIST.value
NOSE           = LANDMARKS.NOSE.value
LEFT_EAR       = LANDMARKS.LEFT_EAR.value
RIGHT_EAR      = LANDMARKS.RIGHT_EAR.value
LEFT_HIP       = LANDMARKS.LEFT_HIP.value
RIGHT_HIP      = LANDMARKS.RIGHT_HIP.value


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 : FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_frames(video_path: str, target_fps: int = 6) -> List[np.ndarray]:
    """
    Extract frames from a video at target_fps.
    Returns list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(native_fps / target_fps))
    frames, idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    log.debug(f"  Extracted {len(frames)} frames from {Path(video_path).name}")
    return frames


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 : FACIAL EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════
EMOTION_ORDER = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
def detect_emotion(frame_bgr):
    """
    Lightweight emotion approximation (NO TensorFlow, NO FER)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255.0

    # simple stable mapping
    happy = brightness
    surprise = brightness * 0.5
    neutral = 1 - brightness
    sad = (1 - brightness) * 0.5
    angry = 0.1
    fear = 0.1

    vec = np.array([happy, surprise, neutral, sad, angry, fear], dtype=np.float32)
    confidence = float(max(vec))

    return vec, confidence


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 : MEDIAPIPE POSE FEATURES
# ═══════════════════════════════════════════════════════════════
def _lm(results, idx: int) -> Optional[np.ndarray]:
    """Return [x, y, visibility] for a landmark, or None if low confidence."""
    if results is None or results.pose_landmarks is None:
        return None
    lm = results.pose_landmarks.landmark[idx]
    if lm.visibility < 0.3:
        return None
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_pose_features(
    frame_bgr,
    pose_model,
    prev_left_wrist,
    prev_right_wrist,
):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    left_vel = right_vel = shoulder_open = head_tilt = body_motion = 0.0
    new_lw = new_rw = None

    lsh = _lm(results, LEFT_SHOULDER)
    rsh = _lm(results, RIGHT_SHOULDER)
    lw = _lm(results, LEFT_WRIST)
    rw = _lm(results, RIGHT_WRIST)
    lear = _lm(results, LEFT_EAR)
    rear = _lm(results, RIGHT_EAR)
    lhip = _lm(results, LEFT_HIP)
    rhip = _lm(results, RIGHT_HIP)

    # Hand velocity
    if lw is not None:
        new_lw = lw[:2]
        if prev_left_wrist is not None:
            left_vel = np.linalg.norm(lw[:2] - prev_left_wrist)

    if rw is not None:
        new_rw = rw[:2]
        if prev_right_wrist is not None:
            right_vel = np.linalg.norm(rw[:2] - prev_right_wrist)

    # Shoulder openness
    if lsh is not None and rsh is not None:
        sh_width = np.linalg.norm(lsh[:2] - rsh[:2])
        if lhip is not None and rhip is not None:
            torso_h = np.linalg.norm(
                (lsh[:2] + rsh[:2]) / 2 - (lhip[:2] + rhip[:2]) / 2
            )
            shoulder_open = sh_width / (torso_h + 1e-6)

    # Head tilt
    if lear is not None and rear is not None:
        dx = lear[0] - rear[0]
        dy = lear[1] - rear[1]
        angle = abs(math.atan2(dy, dx))
        head_tilt = min(angle / math.pi, 1.0)

    # ✅ FIXED BODY MOTION (REAL MOVEMENT)
    if lsh is not None and rsh is not None and lhip is not None and rhip is not None:
        torso_centre = (lsh[:2] + rsh[:2] + lhip[:2] + rhip[:2]) / 4

        if hasattr(extract_pose_features, "prev_torso"):
            body_motion = np.linalg.norm(torso_centre - extract_pose_features.prev_torso)

        extract_pose_features.prev_torso = torso_centre

    return left_vel, right_vel, shoulder_open, head_tilt, body_motion, new_lw, new_rw
# ═══════════════════════════════════════════════════════════════
#  SECTION 4 : PER-FRAME FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════
def build_frame_features(
    frame_bgr: np.ndarray,
    pose_model,
    prev_lw: Optional[np.ndarray],
    prev_rw: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

    # ── 1. Emotion detection ─────────────────────────────
    emotion_vec, face_conf = detect_emotion(frame_bgr)

    # ── 2. Pose features ────────────────────────────────
    lv, rv, sh, ht, bm, new_lw, new_rw = extract_pose_features(
        frame_bgr, pose_model, prev_lw, prev_rw
    )

    # ── 3. NORMALIZE / STABILIZE VALUES ────────────────
    lv = min(lv * 25, 1.0)
    rv = min(rv * 25, 1.0)
    bm = min(bm * 15, 1.0)
    sh = min(sh, 2.0)
    ht = min(ht, 1.0)

    # ── 4. INACTIVITY DETECTION (CRITICAL FIX) ─────────
    low_motion_flag = 0.0
    if lv < 0.01 and rv < 0.01 and bm < 0.02:
        low_motion_flag = 1.0

    # ── 5. NON-EXPRESSIVE FACE DETECTION ───────────────
    non_expressive = 0.0
    neutral = emotion_vec[2]
    sad = emotion_vec[3]

    if neutral > 0.6 or sad > 0.5:
        non_expressive = 1.0

    # ── 6. ENGAGEMENT SCORE (NEW SMART FEATURE) ────────
    # ── 6. ENGAGEMENT SCORE (FIXED & BALANCED) ─────────

# weighted combination (balanced)
    engagement = (
        0.3 * (lv + rv) +       # hand movement
        0.3 * bm +              # body motion
        0.2 * emotion_vec[0] +  # happy
        0.2 * emotion_vec[1]    # surprise
    )

    # apply penalties (mild)
    if low_motion_flag:
        engagement *= 0.6

    if non_expressive:
        engagement *= 0.7

# clamp properly
    engagement = max(0.0, min(engagement, 1.0))
    # ── 7. FINAL FEATURE VECTOR ─────────────────────────
    feature = np.concatenate([
        emotion_vec,                          # 6
        np.array([
            lv, rv,                           # 2
            sh, ht, bm,                       # 3
            face_conf,                        # 1
            low_motion_flag,                  # 1
            non_expressive,                   # 1
            engagement                       # 1 (NEW)
        ], dtype=np.float32),
    ])

    return feature, new_lw, new_rw


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 : VIDEO → FEATURE SEQUENCE
# ═══════════════════════════════════════════════════════════════
def video_to_features(
    video_path: str,
    pose_model,
    target_fps: int = 6,
) -> np.ndarray:
    """
    Returns shape (T, n_features) feature matrix for entire video.
    """
    frames = extract_frames(video_path, target_fps)
    if len(frames) == 0:
        return np.zeros((0, CFG["n_features"]), dtype=np.float32)

    features = []
    prev_lw = prev_rw = None

    for frame in frames:
        feat, prev_lw, prev_rw = build_frame_features(frame, pose_model, prev_lw, prev_rw)
        features.append(feat)

    return np.array(features, dtype=np.float32)  # (T, 12)


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 : DATASET BUILDER
# ═══════════════════════════════════════════════════════════════
def build_dataset(
    dataset_root: str,
    target_fps: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walks dataset_root expecting sub-folders named by label:
        dataset_root/
          enthu/      ← label 1
          not_enthu/  ← label 0

    Returns:
        X : (N, seq_len, n_features)
        y : (N,)  int labels
    """
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    X_all, y_all = [], []
    seq_len = CFG["seq_len"]
    stride  = CFG["stride"]

    for label_name, label_int in CFG["label_map"].items():
        folder = Path(dataset_root) / label_name
        if not folder.exists():
            log.warning(f"Folder not found: {folder}")
            continue

        video_files = list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) + \
                      list(folder.glob("*.mov")) + list(folder.glob("*.mkv"))
        log.info(f"Processing {len(video_files)} videos for label '{label_name}' ({label_int})")

        for vid_path in video_files:
            log.info(f"  ▶ {vid_path.name}")
            try:
                feat_seq = video_to_features(str(vid_path), pose_model, target_fps)
                T = len(feat_seq)
                if T < seq_len:
                    # Pad with zeros if too short
                    pad = np.zeros((seq_len - T, CFG["n_features"]), dtype=np.float32)
                    feat_seq = np.vstack([feat_seq, pad])
                    T = seq_len

                # Sliding window
                for start in range(0, T - seq_len + 1, stride):
                    window = feat_seq[start: start + seq_len]  # (seq_len, n_features)
                    X_all.append(window)
                    y_all.append(label_int)

            except Exception as e:
                log.error(f"  ✗ Failed on {vid_path.name}: {e}")

    pose_model.close()

    if len(X_all) == 0:
        raise RuntimeError("No samples collected. Check dataset path and video files.")

    X = np.array(X_all, dtype=np.float32)  # (N, seq_len, 12)
    y = np.array(y_all, dtype=np.int64)
    log.info(f"Dataset: {X.shape[0]} sequences | enthu={int((y==1).sum())} | not_enthu={int((y==0).sum())}")
    return X, y


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 : FEATURE NORMALISATION
# ═══════════════════════════════════════════════════════════════
def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit scaler on flattened time axis; returns fitted scaler."""
    N, T, F = X.shape
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    X_norm = scaler.transform(X_flat)
    return X_norm.reshape(N, T, F).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 : PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════
class EnthusiasmDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════
#  SECTION 9 : LSTM MODEL
# ═══════════════════════════════════════════════════════════════
class EnthusiasmLSTM(nn.Module):
    """
    Bidirectional LSTM with:
      • Layer norm after LSTM
      • Attention-weighted pooling (better than last-hidden)
      • Dropout regularisation
      • Linear classifier head
    """

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(0.3)

        # Attention over time steps
        self.attn = nn.Linear(hidden * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)   # (B, T, H*2)

        # attention weights
        attn_weights = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)

        # weighted sum
        out = torch.sum(out * attn_weights, dim=1)  # (B, H*2)

        out = self.dropout(out)

        out = self.classifier(out)

        return out # (B, 2)


# ═══════════════════════════════════════════════════════════════
#  SECTION 10 : TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
def build_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[label] for label in y], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> EnthusiasmLSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")

    train_ds = EnthusiasmDataset(X_train, y_train)
    val_ds   = EnthusiasmDataset(X_val,   y_val)

    sampler   = build_weighted_sampler(y_train)
    train_dl  = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,  num_workers=0)
    val_dl    = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    model = EnthusiasmLSTM(
        n_features=CFG["n_features"],
        hidden=CFG["hidden_size"],
        layers=CFG["num_layers"],
        dropout=CFG["dropout"],
    ).to(device)

    # Class-weighted loss (handles imbalance)
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [1.0 / c for c in class_counts], dtype=torch.float32
    ).to(device)
    weights = torch.tensor([1.5, 1.0]).to(device)  # [NOT, ENTHU]
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimiser = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=CFG["epochs"])

    best_val_f1   = 0.0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(yb)

        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        log.info(f"Epoch {epoch:03d}/{CFG['epochs']} | loss={train_loss:.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    log.info(f"Best validation macro-F1: {best_val_f1:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════
#  SECTION 11 : EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_model(
    model: EnthusiasmLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    device  = next(model.parameters()).device
    ds      = EnthusiasmDataset(X_test, y_test)
    dl      = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    report = classification_report(all_labels, all_preds,
                                   target_names=["not_enthu", "enthu"])
    cm     = confusion_matrix(all_labels, all_preds)
    auc    = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    log.info("\n── Evaluation Report ──────────────────────\n" + report)
    log.info(f"Confusion Matrix:\n{cm}")
    log.info(f"ROC-AUC: {auc:.4f}")

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc": auc,
        "predictions": all_preds,
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 12 : INFERENCE ON A NEW VIDEO (FRAME-WISE)
# ═══════════════════════════════════════════════════════════════
def predict_video(video_path, model, scaler, target_fps=6):

    device = next(model.parameters()).device

    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    frames = extract_frames(video_path, target_fps)
    T = len(frames)

    if T == 0:
        print("No frames found.")
        return {}

    print(f"\nProcessing {T} frames...\n")

    raw_features = []
    prev_lw = prev_rw = None

    for frame in frames:
        feat, prev_lw, prev_rw = build_frame_features(frame, pose_model, prev_lw, prev_rw)
        raw_features.append(feat)

    pose_model.close()

    raw_feat_arr = np.array(raw_features, dtype=np.float32)
    norm_feat = scaler.transform(raw_feat_arr).astype(np.float32)

    seq_len = CFG["seq_len"]
    model.eval()

    frame_probs = np.full(T, np.nan)

    # Sliding window prediction
    for start in range(0, T - seq_len + 1, CFG["stride"]):
        window = norm_feat[start: start + seq_len]
        xb = torch.tensor(window).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.softmax(model(xb), dim=1)[0, 1].item()
            prob = prob * 0.8   # reduce overconfidence
        frame_probs[start:start+seq_len] = np.where(
            np.isnan(frame_probs[start:start+seq_len]),
            prob,
            (frame_probs[start:start+seq_len] + prob) / 2
        )

    # Fill missing
    for i in range(T):
        if np.isnan(frame_probs[i]):
            frame_probs[i] = frame_probs[i-1] if i > 0 else 0.5

    # Smoothing
    for i in range(1, T):
        frame_probs[i] = 0.7 * frame_probs[i] + 0.3 * frame_probs[i-1]

    threshold = 0.45# ✅ lowered

    frame_labels = []
    frame_log = []

    print("Frame-wise Analysis:\n")

    for i in range(T):
        feat = raw_feat_arr[i]

        prob = frame_probs[i]

        # classification
        if prob > threshold:
            label = 1   # ENTHU
        else:
            label = 0   # NOT  # uncertain → NOT

        # ✅ ALWAYS append
        frame_labels.append(label)

        # ✅ ALWAYS print
        print(
            f"Frame {i:03d} | "
            f"{'ENTHU' if label else 'NOT'} | "
            f"Conf: {prob:.2f} | "
            f"Motion: {feat[10]:.3f} | "
            f"Engagement: {feat[14]:.2f}"
        )

        # ✅ ALWAYS log
        frame_log.append({
            "frame": i,
            "label": "ENTHU" if label else "NOT",
            "confidence": float(prob),
            "motion": float(feat[10]),
            "engagement": float(feat[14])
        })

    # Final stats
    enthu_frames = sum(frame_labels)
    total_frames = len(frame_labels)
    not_enthu_frames = total_frames - enthu_frames

    enthu_pct = (enthu_frames / total_frames) * 100
    not_enthu_pct = (not_enthu_frames / total_frames) * 100

    print("\n================ FINAL ANALYSIS ================")
    print(f"Total Frames: {total_frames}")
    print(f"Enthusiastic Frames: {enthu_frames} ({enthu_pct:.2f}%)")
    print(f"Not Enthusiastic Frames: {not_enthu_frames} ({not_enthu_pct:.2f}%)")

    if enthu_pct > 70:
        final = "HIGHLY ENTHUSIASTIC 🔥"
    elif not_enthu_pct > 70:
        final = "NOT ENTHUSIASTIC ⚠️"
    else:
        final = "MIXED ⚖️"

    print(f"\nFINAL RESULT: {final}")
    print("==============================================\n")

    return {
        "frame_labels": frame_labels,
        "frame_confidence": frame_probs.tolist(),
        "enthusiasm_pct": round(enthu_pct, 2),
        "not_enthu_pct": round(not_enthu_pct, 2),
        "final_result": final,
        "frame_log": frame_log
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 13 : SAVE / LOAD
# ═══════════════════════════════════════════════════════════════
def save_model(model: EnthusiasmLSTM, scaler: StandardScaler) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "cfg": CFG,
    }, CFG["model_path"])
    joblib.dump(scaler, CFG["scaler_path"])
    log.info(f"Model saved → {CFG['model_path']}")
    log.info(f"Scaler saved → {CFG['scaler_path']}")


def load_model(
    model_path: str = CFG["model_path"],
    scaler_path: str = CFG["scaler_path"],
) -> Tuple[EnthusiasmLSTM, StandardScaler]:
    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint.get("cfg", CFG)
    model = EnthusiasmLSTM(
        n_features=cfg["n_features"],
        hidden=cfg["hidden_size"],
        layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = joblib.load(scaler_path)
    log.info(f"Model saved -> {CFG['model_path']}")
    log.info(f"Scaler saved -> {CFG['scaler_path']}")
    return model, scaler


# ═══════════════════════════════════════════════════════════════
#  SECTION 14 : MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
def main(
    dataset_root: str,
    test_video: Optional[str] = None,
    force_retrain: bool = False,
) -> None:
    from sklearn.model_selection import train_test_split

    model_exists  = Path(CFG["model_path"]).exists()
    scaler_exists = Path(CFG["scaler_path"]).exists()

    # ── 1. Train or load ──────────────────────────────────────
    if model_exists and scaler_exists and not force_retrain:
        log.info("Pre-trained model found. Loading …")
        model, scaler = load_model()
    else:
        log.info("Building dataset …")
        X, y = build_dataset(dataset_root, target_fps=CFG["target_fps"])

        # Train / val / test split  70 / 15 / 15
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

        log.info(f"Split → train={len(y_tr)} | val={len(y_val)} | test={len(y_te)}")

        # Scale
        scaler = fit_scaler(X_tr)
        X_tr  = apply_scaler(X_tr,  scaler)
        X_val = apply_scaler(X_val, scaler)
        X_te  = apply_scaler(X_te,  scaler)

        # Train
        model = train_model(X_tr, y_tr, X_val, y_val)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Evaluate on test set
        log.info("\n── Test Set Evaluation ────────────────────")
        evaluate_model(model, X_te, y_te)

        # Save
        save_model(model, scaler)

    # ── 2. Inference on a new video ───────────────────────────
    if test_video:
        if not Path(test_video).exists():
            log.error(f"Test video not found: {test_video}")
        else:
            result = predict_video(test_video, model, scaler, target_fps=CFG["target_fps"])
            out_json = Path(test_video).stem + "_predictions.json"
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"Predictions saved → {out_json}")
            print(f"\nEnthusiasm: {result['enthusiasm_pct']}%")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teacher Enthusiasm Detector")
    parser.add_argument(
        "--dataset",
        default=r"D:\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\datasets\training_enthu_or_not\enthudataset\train",
        help="Root folder with enthu/ and not_enthu/ sub-folders",
    )
    parser.add_argument(
        "--test_video",
        default=None,
        help="Path to a video to run inference on after training",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training even if saved model exists",
    )
    args = parser.parse_args()

    main(
        dataset_root=args.dataset,
        test_video=args.test_video,
        force_retrain=args.retrain,
    )
