"""
MediSign Assist — Real-time sign language inference (TCN)
==========================================================
Uses MediaPipe Tasks HolisticLandmarker API (mediapipe >= 0.10.3).
The holistic model file is downloaded automatically on first run.

Usage:
    python src/utils/inference.py

Controls:
    Q      — quit
    SPACE  — clear the frame buffer (reset gesture)
"""

import time
import collections
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signs import SIGNS_TARGET, IDX_TO_SIGN, NUM_CLASSES  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_T      = 32    # frames expected by the model

BUFFER_MAX     = 64   # max frames kept in the sliding window
PRED_STRIDE    = 16   # run inference every N new frames (once buffer is full enough)
MIN_FRAMES     = TARGET_T
CONF_THRESHOLD = 0.40

VOTE_WINDOW    = 7    # majority vote over last N raw predictions
GRACE_FRAMES   = 12   # frames to wait before clearing buffer when hands disappear
DETECTION_COOLDOWN = 30  # frames (~1 s à 30 fps) pendant lesquels on fige l'affichage
                          # et on arrête de collecter, après une détection confiante

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TCN_MODEL_PATH = MODELS_DIR / "tcn_medisign_final.pth"

HOLISTIC_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)
HOLISTIC_MODEL_PATH = MODELS_DIR / "holistic_landmarker.task"

# ── MediaPipe model download ───────────────────────────────────────────────────

def ensure_holistic_model() -> None:
    if HOLISTIC_MODEL_PATH.exists():
        return
    print(f"[INFO] Téléchargement du modèle MediaPipe Holistic (~30 MB)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
    print(f"[OK]   Modèle sauvegardé : {HOLISTIC_MODEL_PATH.name}")

# ── TCN model (must match training architecture) ───────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding      = (kernel_size - 1) * dilation // 2
        self.conv1   = nn.Conv1d(in_channels,  out_channels, kernel_size,
                                 padding=padding, dilation=dilation)
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.conv2   = nn.Conv1d(out_channels, out_channels, kernel_size,
                                 padding=padding, dilation=dilation)
        self.bn2     = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, in_channels: int = 225, num_classes: int = 20,
                 hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=1,  dropout=dropout),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=2,  dropout=dropout),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=4,  dropout=dropout),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=8,  dropout=dropout),
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


def load_model(device: torch.device) -> TCN:
    model   = TCN(in_channels=225, num_classes=NUM_CLASSES, hidden=128, dropout=0.3)
    payload = torch.load(TCN_MODEL_PATH, map_location=device, weights_only=False)
    state   = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[OK]   TCN chargé — {TCN_MODEL_PATH.name}")
    if "metadata" in payload:
        m = payload["metadata"]
        print(f"       test accuracy={m.get('test_accuracy','?')}  "
              f"macro-F1={m.get('test_macro_f1','?')}")
    return model


# ── Landmark extraction ────────────────────────────────────────────────────────

def extract_keypoints(result: mp_vision.HolisticLandmarkerResult) -> np.ndarray:
    """
    Build a (225,) feature vector from one HolisticLandmarkerResult.
    Layout (matches training): 33 pose × 3 + 21 left_hand × 3 + 21 right_hand × 3 = 225
    Missing landmarks are filled with zeros.
    """
    def lm_to_arr(landmarks, n: int) -> np.ndarray:
        if not landmarks:
            return np.zeros((n, 3), dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                        dtype=np.float32)[:n]

    pose  = lm_to_arr(result.pose_landmarks,       33)  # (33, 3)
    lhand = lm_to_arr(result.left_hand_landmarks,  21)  # (21, 3)
    rhand = lm_to_arr(result.right_hand_landmarks, 21)  # (21, 3)
    return np.concatenate([pose.flatten(), lhand.flatten(), rhand.flatten()])  # (225,)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def resample_to_target(buffer: np.ndarray, target_T: int = TARGET_T) -> np.ndarray:
    """buffer: (T, 225)  →  (225, target_T)  channel-first for Conv1d"""
    T = len(buffer)
    if T == target_T:
        return buffer.T.astype(np.float32)
    idx       = np.linspace(0, T - 1, target_T)
    lo        = np.floor(idx).astype(int).clip(0, T - 1)
    hi        = np.ceil(idx).astype(int).clip(0, T - 1)
    alpha     = (idx - lo)[:, None]
    resampled = ((1 - alpha) * buffer[lo] + alpha * buffer[hi]).astype(np.float32)
    return resampled.T  # (225, target_T)


# ── TCN inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_top3(model: TCN, coords: np.ndarray, device: torch.device
                 ) -> list[tuple[str, float]]:
    """
    coords: (225, T)
    Returns list of (sign_name, confidence) sorted by confidence desc, top-3.
    """
    x      = torch.from_numpy(coords).unsqueeze(0).to(device)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)[0]
    top3_idx = probs.argsort(descending=True)[:3].tolist()
    return [(IDX_TO_SIGN[i], probs[i].item()) for i in top3_idx]


# ── Skeleton drawing (manual, no mp.solutions dependency) ─────────────────────

_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
]
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _draw_landmarks(frame: np.ndarray, landmarks, connections: list,
                    line_color: tuple, dot_color: tuple = (255, 255, 255)) -> None:
    if not landmarks:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, dot_color, -1, cv2.LINE_AA)


# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray,
                 voted_sign: str | None,
                 top3: list[tuple[str, float]],
                 buf_len: int,
                 collecting: bool,
                 cooldown: int = 0) -> None:
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 78), (20, 20, 20), -1)

    # Buffer progress bar
    bar_w = int((min(buf_len, BUFFER_MAX) / BUFFER_MAX) * 180)
    cv2.rectangle(frame, (10, 8),  (190, 22), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 8),  (10 + bar_w, 22), (0, 200, 80), -1)

    if cooldown > 0:
        status_color = (0, 200, 255)
        status_label = f"Detecte ! Prochain geste dans {cooldown}..."
    elif collecting:
        status_color = (0, 220, 80)
        status_label = f"Buffer {buf_len}/{BUFFER_MAX}"
    else:
        status_color = (80, 80, 200)
        status_label = "En attente de mains..."
    cv2.putText(frame, status_label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    # ── Voted prediction (main label, centre-haut) ────────────────────────────
    if voted_sign is not None:
        cv2.putText(frame, voted_sign, (w // 2 - 80, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 180), 2, cv2.LINE_AA)

    # ── Top-3 panel (bas-droite) ─────────────────────────────────────────────
    panel_x  = w - 220
    panel_y0 = h - 10 - 3 * 28
    cv2.rectangle(frame, (panel_x - 8, panel_y0 - 8),
                  (w - 4, h - 4), (25, 25, 25), -1)

    for rank, (sign, conf) in enumerate(top3):
        y       = panel_y0 + rank * 28
        alpha   = 1.0 - rank * 0.3          # dim lower ranks
        c_val   = int(255 * alpha)
        color   = (0, c_val, int(c_val * 0.7))
        bar_len = int(conf * 100)
        cv2.rectangle(frame, (panel_x, y + 4),
                      (panel_x + bar_len, y + 14), (0, 80, 50), -1)
        label   = f"#{rank+1} {sign:<10} {conf*100:.0f}%"
        cv2.putText(frame, label, (panel_x, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ── Bottom hint ───────────────────────────────────────────────────────────
    cv2.putText(frame, "Q: quitter   ESPACE: reinitialiser", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ensure_holistic_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    model = load_model(device)

    base_options = mp_tasks.BaseOptions(model_asset_path=str(HOLISTIC_MODEL_PATH))
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra (index 0).")

    # State
    buffer: list[np.ndarray]       = []
    frames_since_last_pred: int    = 0
    grace_counter: int             = 0   # frames since hands disappeared
    detection_cooldown: int        = 0   # frames restants avant reset post-détection
    vote_history: collections.deque = collections.deque(maxlen=VOTE_WINDOW)
    voted_sign: str | None         = None
    last_top3: list                = []
    t0 = time.time()

    print("[INFO] Fenetre ouverte — faites un signe devant la camera.")

    with mp_vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame non reçue, arrêt.")
                break

            timestamp_ms = int((time.time() - t0) * 1000)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands_visible = bool(result.left_hand_landmarks or result.right_hand_landmarks)

            # ── Buffer management ────────────────────────────────────────────
            if hands_visible:
                grace_counter = 0

                # Cooldown post-détection : on affiche le résultat, on ne collecte pas
                if detection_cooldown > 0:
                    detection_cooldown -= 1
                    if detection_cooldown == 0:
                        buffer.clear()
                        vote_history.clear()
                        frames_since_last_pred = 0
                        print("[INFO] Buffer réinitialisé après détection.")
                else:
                    buffer.append(extract_keypoints(result))
                    if len(buffer) > BUFFER_MAX:
                        buffer.pop(0)
                    frames_since_last_pred += 1

                    if (len(buffer) >= MIN_FRAMES and
                            frames_since_last_pred >= PRED_STRIDE):
                        coords   = resample_to_target(np.array(buffer))
                        top3     = predict_top3(model, coords, device)
                        last_top3 = top3
                        best_sign = top3[0][0]
                        vote_history.append(best_sign)
                        # Majority vote over recent predictions
                        voted_sign = collections.Counter(vote_history).most_common(1)[0][0]
                        frames_since_last_pred = 0
                        print(f"  → {top3[0][0]} ({top3[0][1]*100:.1f}%)  "
                              f"[vote: {voted_sign}]  "
                              f"top3: {[(s, f'{c*100:.0f}%') for s, c in top3]}")
                        # Auto-reset après détection confiante
                        if top3[0][1] >= CONF_THRESHOLD:
                            detection_cooldown = DETECTION_COOLDOWN
            else:
                grace_counter += 1
                if grace_counter >= GRACE_FRAMES:
                    # Run a final prediction before clearing
                    if len(buffer) >= MIN_FRAMES and frames_since_last_pred > 0:
                        coords    = resample_to_target(np.array(buffer))
                        top3      = predict_top3(model, coords, device)
                        last_top3 = top3
                        best_sign = top3[0][0]
                        vote_history.append(best_sign)
                        voted_sign = collections.Counter(vote_history).most_common(1)[0][0]
                        print(f"  → {top3[0][0]} ({top3[0][1]*100:.1f}%)  [fin de geste]")
                    buffer.clear()
                    vote_history.clear()
                    frames_since_last_pred = 0
                    detection_cooldown = 0

            # ── Draw skeleton ────────────────────────────────────────────────
            _draw_landmarks(frame, result.pose_landmarks,
                            _POSE_CONNECTIONS, (80, 200, 80))
            _draw_landmarks(frame, result.left_hand_landmarks,
                            _HAND_CONNECTIONS, (121, 44, 250))
            _draw_landmarks(frame, result.right_hand_landmarks,
                            _HAND_CONNECTIONS, (245, 117, 66))

            draw_overlay(frame, voted_sign, last_top3, len(buffer), hands_visible,
                         cooldown=detection_cooldown)
            cv2.imshow("MediSign — TCN Inference", frame)

            # ── Keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                buffer.clear()
                vote_history.clear()
                voted_sign = None
                last_top3  = []
                frames_since_last_pred = 0
                grace_counter          = 0
                print("[INFO] Buffer réinitialisé.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()
