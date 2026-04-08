"""
MediSign Assist — Pose re-extraction
=====================================
Relit chaque vidéo du dataset et ré-extrait les landmarks MediaPipe
avec le nouveau HolisticLandmarker (Tasks API) pour correspondre
exactement au modèle utilisé en inférence temps réel.

Les fichiers .npy produits remplacent les poses téléchargées depuis LSFB
et ont le même format :
    pose/      {id}.npy  →  (T, 33, 3)
    left_hand/ {id}.npy  →  (T, 21, 3)
    right_hand/{id}.npy  →  (T, 21, 3)

Usage :
    python src/utils/extract_poses.py              # tout le dataset
    python src/utils/extract_poses.py --skip-existing   # saute ce qui est déjà fait
    python src/utils/extract_poses.py --limit 100  # test sur 100 vidéos
"""

import argparse
import sys
import urllib.request
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signs import SIGNS_TARGET  # noqa: E402

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent  # src/
DATASET_ROOT = ROOT / "dataset"
VIDEOS_DIR   = DATASET_ROOT / "videos"
POSES_ROOT   = DATASET_ROOT / "poses"
MODELS_DIR   = ROOT / "models"

HOLISTIC_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)
HOLISTIC_MODEL_PATH = MODELS_DIR / "holistic_landmarker.task"

# ── Download model ─────────────────────────────────────────────────────────────

def ensure_holistic_model() -> None:
    if HOLISTIC_MODEL_PATH.exists():
        return
    print("[INFO] Téléchargement du modèle MediaPipe Holistic (~30 MB)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
    print(f"[OK]   {HOLISTIC_MODEL_PATH.name}")

# ── Per-video extraction ───────────────────────────────────────────────────────

def _lm_to_arr(landmarks, n: int) -> np.ndarray:
    """Convert a landmark list to (n, 3) float32 array, zero-filled if absent."""
    if not landmarks:
        return np.zeros((n, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32)[:n]


def extract_video(video_path: Path, landmarker: mp_vision.HolisticLandmarker
                  ) -> dict[str, np.ndarray] | None:
    """
    Process one video and return dict of arrays:
        "pose"       (T, 33, 3)
        "left_hand"  (T, 21, 3)
        "right_hand" (T, 21, 3)
    Returns None if the video cannot be opened or has 0 frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    pose_frames, lhand_frames, rhand_frames = [], [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx / fps * 1000)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result       = landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_frames.append(_lm_to_arr(result.pose_landmarks,       33))
        lhand_frames.append(_lm_to_arr(result.left_hand_landmarks, 21))
        rhand_frames.append(_lm_to_arr(result.right_hand_landmarks, 21))
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return None

    return {
        "pose":       np.stack(pose_frames,  axis=0),   # (T, 33, 3)
        "left_hand":  np.stack(lhand_frames, axis=0),   # (T, 21, 3)
        "right_hand": np.stack(rhand_frames, axis=0),   # (T, 21, 3)
    }


def save_arrays(instance_id: str, arrays: dict[str, np.ndarray]) -> None:
    for subdir, arr in arrays.items():
        out_path = POSES_ROOT / subdir / f"{instance_id}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="Saute les instances dont les .npy existent déjà")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre max de vidéos à traiter (debug)")
    args = parser.parse_args()

    ensure_holistic_model()

    # Load instance list
    instances_csv = DATASET_ROOT / "instances.csv"
    df = pd.read_csv(instances_csv)

    df = df[df["sign"].isin(SIGNS_TARGET)].reset_index(drop=True)

    # Filter to videos that actually exist on disk
    df["video_path"] = df["id"].apply(lambda i: VIDEOS_DIR / f"{i}.mp4")
    df = df[df["video_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    print(f"[INFO] {len(df)} vidéos trouvées sur {len(SIGNS_TARGET)} signes cibles")

    if args.skip_existing:
        def already_done(instance_id):
            return (POSES_ROOT / "pose" / f"{instance_id}.npy").exists()
        before = len(df)
        df = df[~df["id"].apply(already_done)].reset_index(drop=True)
        print(f"[INFO] {before - len(df)} déjà traités, {len(df)} restants")

    if args.limit is not None:
        df = df.iloc[:args.limit]
        print(f"[INFO] Limite : {args.limit} vidéos")

    # Build landmarker (VIDEO mode — one instance per create call)
    base_options = mp_tasks.BaseOptions(model_asset_path=str(HOLISTIC_MODEL_PATH))
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    ok, skipped = 0, 0

    # We must create a fresh landmarker per video because VIDEO mode
    # requires monotonically increasing timestamps (reset between videos).
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extraction"):
        instance_id = row["id"]
        video_path  = row["video_path"]

        with mp_vision.HolisticLandmarker.create_from_options(options) as landmarker:
            arrays = extract_video(video_path, landmarker)

        if arrays is None:
            tqdm.write(f"[WARN] Vidéo ignorée (vide ou illisible) : {video_path.name}")
            skipped += 1
            continue

        save_arrays(instance_id, arrays)
        ok += 1

    print(f"\n[DONE] {ok} extraits  |  {skipped} ignorés")
    print(f"       Poses sauvegardées dans : {POSES_ROOT}")


if __name__ == "__main__":
    main()
