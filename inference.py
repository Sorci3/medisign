# inference.py — Inférence en temps réel depuis la webcam
#
# Fonctionnement :
#   - Collecte 60 frames (~2s à 30fps) dans un buffer
#   - Prédit le signe à la fin du cycle et reset automatiquement
#   - Affiche le top-5 des candidats avec barres de confiance
#
# Contrôles :
#   q : quitter
#   c : reset manuel du buffer

import os
import sys
import json
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import SPOTER
from utils import normalize_sequence

# ─── Config ───────────────────────────────────────────────────────────────────
SEUIL_CONFIANCE = 0.70   # seuil minimum pour afficher une prédiction
WEBCAM_ID       = 0
CAPTURE_FRAMES  = 60     # frames par cycle (~2s à 30fps)
FEATURE_SIZE    = 225
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR   = os.path.join(_HERE, "models", "spoter")
MP_MODEL_DIR = os.path.join(_HERE, "models", "mediapipe")
os.makedirs(MP_MODEL_DIR, exist_ok=True)


# ─── Chargement du modèle SPOTER ──────────────────────────────────────────────

def load_model():
    ckpt_path    = os.path.join(MODELS_DIR, "best_model.pt")
    meta_path    = os.path.join(MODELS_DIR, "meta.json")
    classes_path = os.path.join(MODELS_DIR, "classes.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Modele introuvable : {ckpt_path}\n"
            "Lancez d'abord : python train.py")

    with open(meta_path)    as f: meta    = json.load(f)
    with open(classes_path) as f: classes = json.load(f)

    model = SPOTER(
        num_classes        = len(classes),
        feature_size       = FEATURE_SIZE,
        hidden_dim         = meta["hidden_dim"],
        nhead              = meta["nhead"],
        num_encoder_layers = meta["num_encoder_layers"],
        num_decoder_layers = meta["num_decoder_layers"],
        dim_feedforward    = meta["dim_feedforward"],
        dropout            = meta["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"Modele charge : {len(classes)} classes — {DEVICE}")
    return model, classes, meta["max_len"]


# ─── Téléchargement des modèles MediaPipe ─────────────────────────────────────

POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def download_if_needed(url, path):
    if not os.path.exists(path):
        print(f"Telechargement {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)

pose_path = os.path.join(MP_MODEL_DIR, "pose_landmarker_lite.task")
hand_path = os.path.join(MP_MODEL_DIR, "hand_landmarker.task")
download_if_needed(POSE_URL, pose_path)
download_if_needed(HAND_URL, hand_path)


# ─── Détecteurs MediaPipe ─────────────────────────────────────────────────────

pose_det = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_path),
        running_mode=mp_vision.RunningMode.IMAGE, num_poses=1))

hand_det = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_path),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=2))

N_POSE, N_HAND = 33, 21


def get_landmarks(rgb):
    """Extrait pose + mains depuis une frame RGB → vecteur (225,)."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pr   = pose_det.detect(img)
    pose = (np.array([[lm.x, lm.y, lm.z] for lm in pr.pose_landmarks[0]], dtype=np.float32).flatten()
            if pr.pose_landmarks else np.zeros(N_POSE * 3, dtype=np.float32))

    hr    = hand_det.detect(img)
    left  = np.zeros(N_HAND * 3, dtype=np.float32)
    right = np.zeros(N_HAND * 3, dtype=np.float32)
    if hr.hand_landmarks:
        for i, lms in enumerate(hr.hand_landmarks):
            arr  = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
            side = hr.handedness[i][0].category_name
            if side == "Left":  left  = arr
            else:               right = arr

    return np.concatenate([pose, left, right])


# ─── Prédiction sur le buffer ─────────────────────────────────────────────────

model, classes, MAX_LEN = load_model()

def predict(buffer):
    """
    buffer : liste de vecteurs (225,) — une entrée par frame
    Applique la normalisation Bohacek sur la séquence entière,
    padde à MAX_LEN et retourne (label, confiance, top5).
    """
    seq = np.array(buffer[-MAX_LEN:], dtype=np.float32)   # (T, 225)
    seq = normalize_sequence(seq)

    inp = np.zeros((1, MAX_LEN, FEATURE_SIZE), dtype=np.float32)
    inp[0, MAX_LEN - len(seq):] = seq

    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(inp).to(DEVICE))[0], dim=0).cpu().numpy()

    top5  = [(classes[i], float(probs[i])) for i in probs.argsort()[::-1][:5]]
    conf  = top5[0][1]
    label = top5[0][0] if conf >= SEUIL_CONFIANCE else "..."
    return label, conf, top5


# ─── Boucle principale ────────────────────────────────────────────────────────

cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam"); exit()

buffer     = []
last_label = "..."
last_conf  = 0.0
last_top5  = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lm  = get_landmarks(rgb)
    buffer.append(lm)

    # Affichage des points de pose
    h, w = frame.shape[:2]
    for i in range(N_POSE):
        x, y = int(lm[i*3]*w), int(lm[i*3+1]*h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    for i in range(N_HAND):
        off = N_POSE * 3
        for j, color in [(i, (255, 44, 250)), (i + N_HAND, (245, 117, 66))]:
            x, y = int(lm[off + j*3]*w), int(lm[off + j*3+1]*h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 3, color, -1)

    # Prédiction automatique quand le buffer est plein
    if len(buffer) >= CAPTURE_FRAMES:
        last_label, last_conf, last_top5 = predict(buffer)
        buffer.clear()

    # HUD
    hud_h   = 50 + len(last_top5) * 28 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color = (0, 220, 0) if last_conf >= SEUIL_CONFIANCE else (0, 165, 255)
    cv2.putText(frame, f"{last_label}  ({last_conf*100:.0f}%)",
                (20, h - hud_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    for rank, (lbl, prob) in enumerate(last_top5):
        y_pos   = h - hud_h + 60 + rank * 28
        bar_len = int(prob * (w // 3))
        c       = (0, 220, 0) if rank == 0 and prob >= SEUIL_CONFIANCE else (180, 180, 180)
        cv2.rectangle(frame, (20, y_pos - 14), (20 + bar_len, y_pos - 2), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, y_pos - 14), (20 + bar_len, y_pos - 2), c, 1)
        cv2.putText(frame, f"{rank+1}. {lbl:<20s} {prob*100:5.1f}%",
                    (26 + w // 3, y_pos - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

    # Barre de progression du buffer
    bar_w = int(len(buffer) / CAPTURE_FRAMES * (w - 40))
    cv2.rectangle(frame, (20, h - 10), (20 + bar_w, h - 2), (100, 200, 255), -1)
    cv2.putText(frame, "q:quitter  c:reset", (20, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("MediSign — SPOTER", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    elif key == ord("c"):
        buffer.clear(); last_label = "..."; last_conf = 0.0; last_top5 = []

cap.release()
cv2.destroyAllWindows()
pose_det.close()
hand_det.close()
