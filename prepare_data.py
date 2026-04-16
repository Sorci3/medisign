# prepare_data.py — Préparation des données LSFB en 3 étapes
#
# Usage :
#   python prepare_data.py                    # exécute les 3 étapes dans l'ordre
#   python prepare_data.py --step download    # étape 1 : télécharge le dataset
#   python prepare_data.py --step index       # étape 2 : génère index.csv
#   python prepare_data.py --step landmarks   # étape 3 : fusionne les poses en .npy
#
# Les données sont téléchargées dans dataset_path défini dans config.json.

import os
import json
import argparse
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from lsfb_dataset import Downloader

# ─── Config ───────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "config.json")) as f:
    CFG = json.load(f)

DATASET  = os.path.realpath(os.path.join(_HERE, CFG["dataset_path"]))
BASE_URL = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"

# 20 signes médicaux ciblés
SIGNS = [
    "OUI", "NON", "ENCEINTE", "COMPRENDRE", "SOUFFRIR", "CHAUD", "FROID",
    "HOPITAL", "MEDECIN", "APPELER", "MALADE.VENTRE", "PAS.SOUFFLER",
    "RESPIRER", "TOMBER", "BOIRE", "DORMIR", "MANGER", "FORT", "FAIBLE", "VITE",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Téléchargement du dataset
# ═══════════════════════════════════════════════════════════════════════════════

def step_download():
    """
    Télécharge les vidéos et poses des 20 signes depuis LSFB ISOL v2.
    Reprend automatiquement si interrompu (skip_existing_files=True).
    """
    os.makedirs(DATASET, exist_ok=True)
    csv_path = os.path.join(DATASET, "instances.csv")

    # Télécharge instances.csv si absent
    if not os.path.exists(csv_path):
        print("Téléchargement de instances.csv...")
        r = requests.get(f"{BASE_URL}/instances.csv", timeout=60)
        with open(csv_path, "wb") as f:
            f.write(r.content)

    df         = pd.read_csv(csv_path)
    df_filtered = df[df["sign"].isin(SIGNS)].reset_index(drop=True)
    our_ids    = set(df_filtered["id"].tolist())

    print(f"Instances a telecharger : {len(df_filtered)} "
          f"({df_filtered['sign'].nunique()} signes)")

    # Utilise le Downloader LSFB officiel
    downloader = Downloader(
        dataset="isol", destination=DATASET, splits=["all"],
        include_videos=False, include_cleaned_poses=False,
        include_raw_poses=False, skip_existing_files=True,
        landmarks=["pose", "left_hand", "right_hand"],  # face inutile (FEATURE_SIZE=225)
    )
    downloader.download()  # télécharge les métadonnées (splits JSON)

    downloader.instances            = [i for i in downloader.instances if i in our_ids]
    downloader.include_cleaned_poses = True
    downloader._download_files(
        downloader._get_pose_origins(),
        title=f"Poses ({len(downloader.instances)} instances)"
    )
    print("\nTelechargement termine.")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Génération de l'index
# ═══════════════════════════════════════════════════════════════════════════════

def step_index():
    """
    Génère index.csv en ne gardant que les instances dont les poses sont présentes sur disque.
    Colonnes : id, sign, label.
    """
    csv_path   = os.path.join(DATASET, "instances.csv")
    poses_dir  = os.path.join(DATASET, "poses", "pose")
    index_path = os.path.join(DATASET, "index.csv")

    if not os.path.exists(csv_path):
        print("instances.csv introuvable. Lancez d'abord : python prepare_data.py --step download")
        return

    df          = pd.read_csv(csv_path)
    df_filtered = df[df["sign"].isin(SIGNS)].copy()

    # Garde uniquement les instances dont la pose est présente sur disque
    available = {os.path.splitext(f)[0] for f in os.listdir(poses_dir) if f.endswith(".npy")}
    df_filtered = df_filtered[df_filtered["id"].isin(available)].reset_index(drop=True)

    label_map         = {s: i for i, s in enumerate(sorted(SIGNS))}
    df_filtered["label"] = df_filtered["sign"].map(label_map)
    df_filtered[["id", "sign", "label"]].to_csv(index_path, index=False)

    print(f"index.csv cree : {len(df_filtered)} entrees, {df_filtered['sign'].nunique()} classes")
    for sign, count in df_filtered["sign"].value_counts().sort_index().items():
        print(f"  {label_map[sign]}  {sign:<25s} {count} instances")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Extraction des landmarks
# ═══════════════════════════════════════════════════════════════════════════════

def merge_poses(vid_id, poses_dir):
    """Fusionne pose + left_hand + right_hand en un seul vecteur (T, 225)."""
    pose  = np.load(os.path.join(poses_dir, "pose",       vid_id + ".npy"))  # (T, 33, 3)
    left  = np.load(os.path.join(poses_dir, "left_hand",  vid_id + ".npy"))  # (T, 21, 3)
    right = np.load(os.path.join(poses_dir, "right_hand", vid_id + ".npy"))  # (T, 21, 3)
    T = pose.shape[0]
    if T == 0:
        return np.empty((0, 225), dtype=np.float32)
    return np.concatenate([
        pose.reshape(T, -1),
        left.reshape(T, -1),
        right.reshape(T, -1),
    ], axis=1).astype(np.float32)


def step_landmarks():
    """
    Fusionne les poses LSFB (pose + mains) en fichiers .npy de forme (T, 225).
    Un fichier par vidéo, stocké dans DATASET/landmarks/.
    """
    index_path   = os.path.join(DATASET, "index.csv")
    poses_dir    = os.path.join(DATASET, "poses")
    landmark_dir = os.path.join(DATASET, "landmarks")
    os.makedirs(landmark_dir, exist_ok=True)

    if not os.path.exists(index_path):
        print("index.csv introuvable. Lancez d'abord : python prepare_data.py --step index")
        return

    df      = pd.read_csv(index_path)
    skipped = errors = 0

    print(f"Fusion des poses pour {len(df)} videos -> {landmark_dir}")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(landmark_dir, row["id"] + ".npy")
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            feat = merge_poses(row["id"], poses_dir)
        except FileNotFoundError as e:
            print(f"  pose manquante : {row['id']} ({e})")
            errors += 1
            continue
        if len(feat) == 0:
            errors += 1
            continue
        np.save(out_path, feat)

    print(f"Fini : {len(df) - skipped - errors} merges, {skipped} deja presents, {errors} erreurs")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3b — Téléchargement des poses pour le pré-entraînement (dataset complet)
# ═══════════════════════════════════════════════════════════════════════════════

def step_pretrain_download(min_instances=50):
    """
    Télécharge les poses de tout le dataset LSFB (452 classes, ~60k instances)
    nécessaires au pré-entraînement. Filtre les signes avec moins de min_instances.
    """
    csv_path = os.path.join(DATASET, "instances.csv")
    if not os.path.exists(csv_path):
        print("instances.csv introuvable. Lancez d'abord : python prepare_data.py --step download")
        return

    df     = pd.read_csv(csv_path)
    counts = df["sign"].value_counts()
    df     = df[df["sign"].isin(counts[counts >= min_instances].index)].reset_index(drop=True)
    all_ids = set(df["id"].tolist())

    print(f"Instances a telecharger : {len(df)} ({df['sign'].nunique()} signes, seuil >= {min_instances})")

    downloader = Downloader(
        dataset="isol", destination=DATASET, splits=["all"],
        include_videos=False, include_cleaned_poses=False,
        include_raw_poses=False, skip_existing_files=True,
        landmarks=["pose", "left_hand", "right_hand"],
    )
    downloader.download()  # métadonnées (splits JSON)

    downloader.instances             = [i for i in downloader.instances if i in all_ids]
    downloader.include_cleaned_poses = True
    downloader._download_files(
        downloader._get_pose_origins(),
        title=f"Poses pretrain ({len(downloader.instances)} instances)"
    )
    print("\nTelechargement pretrain termine.")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Landmarks pré-entraînement (dataset complet, normalisés)
# ═══════════════════════════════════════════════════════════════════════════════

def step_pretrain_landmarks(min_instances=50):
    """
    Génère les landmarks normalisés pour tout le dataset LSFB.
    Résultat : DATASET/pretrain_landmarks/<id>.npy — forme (T, 225), déjà normalisé.
    """
    import sys
    sys.path.insert(0, _HERE)
    from utils import normalize_sequence

    csv_path      = os.path.join(DATASET, "instances.csv")
    poses_dir     = os.path.join(DATASET, "poses")
    landmark_dir  = os.path.join(DATASET, "pretrain_landmarks")
    os.makedirs(landmark_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print("instances.csv introuvable. Lancez d'abord : python prepare_data.py --step download")
        return

    df     = pd.read_csv(csv_path)
    counts = df["sign"].value_counts()
    df     = df[df["sign"].isin(counts[counts >= min_instances].index)].reset_index(drop=True)
    print(f"Instances a traiter : {len(df)} ({df['sign'].nunique()} signes, seuil >= {min_instances})")

    skipped = errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(landmark_dir, row["id"] + ".npy")
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            feat = merge_poses(row["id"], poses_dir)
            feat = np.nan_to_num(normalize_sequence(feat))
        except FileNotFoundError:
            errors += 1
            continue
        if len(feat) == 0:
            errors += 1
            continue
        np.save(out_path, feat)

    print(f"Fini : {len(df) - skipped - errors} generes, {skipped} deja presents, {errors} erreurs")
    print(f"Dossier : {landmark_dir}")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Preparation des donnees LSFB")
    parser.add_argument("--step",
                        choices=["download", "index", "landmarks",
                                 "pretrain-download", "pretrain-landmarks", "all"],
                        default="all", help="Etape a executer (defaut: all)")
    args = parser.parse_args()

    if args.step in ("all", "download"):
        print("=== Etape 1 : Telechargement (20 signes) ===")
        step_download()
    if args.step in ("all", "index"):
        print("\n=== Etape 2 : Index ===")
        step_index()
    if args.step in ("all", "landmarks"):
        print("\n=== Etape 3 : Landmarks (20 signes) ===")
        step_landmarks()
    if args.step == "pretrain-download":
        print("\n=== Etape 3b : Telechargement poses pretrain (dataset complet) ===")
        step_pretrain_download()
    if args.step == "pretrain-landmarks":
        print("\n=== Etape 4 : Landmarks pretrain (dataset complet) ===")
        step_pretrain_landmarks()


if __name__ == "__main__":
    main()
