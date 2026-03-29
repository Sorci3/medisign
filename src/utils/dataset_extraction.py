"""
MediSign Assist - Dataset Downloader
=====================================
Télécharge un sous-ensemble ciblé du dataset LSFB ISOL
(20 signes médicaux d'urgence)

Prérequis :
    pip install requests pandas tqdm lsfb_dataset

Utilisation :
    python dataset_extraction.py
    Modifier DESTINATION si nécessaire.
"""

import os
import json
import requests
import pandas as pd
from lsfb_dataset import Downloader

# =============================================================================
# CONFIGURATION
# =============================================================================

DESTINATION = "src/dataset"

MY_SIGNS_EXACT = [
    "SOUFFRIR", "AIDER",    "FORT",     "MALADE",   "COEUR",
    "TETE",     "MORT",     "DOS",      "VENTRE",   "FROID",
    "JAMBE",    "JAMBES",   "RESPIRER", "ACCIDENT", "FAIBLE",
    "ENCEINTE", "DIABETE",  "BRAS",     "DOSSIER",  "EFFORT"
]

DOWNLOAD_VIDEOS = True
DOWNLOAD_POSES  = True   # poses nettoyées (interpolées + lissées)
DOWNLOAD_POSES_RAW = False

# =============================================================================

BASE_URL = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"  # URL corrigée
CSV_URL  = f"{BASE_URL}/instances.csv"

os.makedirs(DESTINATION, exist_ok=True)

# --- Étape 1 : Télécharger instances.csv ---
csv_path = os.path.join(DESTINATION, "instances.csv")

def download_and_validate_csv() -> pd.DataFrame:
    print("Téléchargement de instances.csv...")
    r = requests.get(CSV_URL, timeout=60)
    # Le serveur renvoie du HTML si l'URL est mauvaise
    if "text/html" in r.headers.get("Content-Type", ""):
        raise ValueError(f"URL incorrecte — le serveur a renvoyé du HTML.\nURL testée : {CSV_URL}")
    with open(csv_path, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(csv_path)
    if 'sign' not in df.columns:
        os.remove(csv_path)
        raise ValueError(f"CSV invalide. Colonnes reçues : {df.columns.tolist()}")
    print(f"✓ instances.csv valide ({len(df)} instances)\n")
    return df

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if 'sign' not in df.columns:
        print("⚠ instances.csv corrompu, re-téléchargement...")
        df = download_and_validate_csv()
    else:
        print(f"✓ instances.csv déjà présent ({len(df)} instances)\n")
else:
    df = download_and_validate_csv()

# --- Étape 2 : Filtrage des IDs qui nous intéressent ---
df_filtered = df[df['sign'].isin(MY_SIGNS_EXACT)].reset_index(drop=True)

found_signs   = sorted(df_filtered['sign'].unique().tolist())
missing_signs = sorted(set(MY_SIGNS_EXACT) - set(found_signs))

print(f"Signes trouvés ({len(found_signs)}/{len(MY_SIGNS_EXACT)}) : {found_signs}")
if missing_signs:
    print(f"⚠ Signes absents du dataset : {missing_signs}")
print(f"Instances à télécharger : {len(df_filtered)}\n")

our_ids = set(df_filtered['id'].tolist())

# --- Étape 3 : Downloader officiel — télécharge les métadonnées (JSON splits, etc.) ---
print("Téléchargement des métadonnées LSFB (splits JSON)...")
downloader = Downloader(
    dataset='isol',
    destination=DESTINATION,
    splits=['all'],
    include_videos=False,
    include_cleaned_poses=False,
    include_raw_poses=False,
    skip_existing_files=True,
)
# On lance download() uniquement pour les métadonnées
# (poses et vidéos désactivées, on les gère après)
downloader.download()
print("✓ Métadonnées téléchargées\n")

# --- Étape 4 : Injecter nos IDs filtrés et télécharger poses + vidéos ---
# Le Downloader stocke les instances comme une liste de strings (IDs)
downloader.instances = [inst for inst in downloader.instances if inst in our_ids]
print(f"Instances injectées : {len(downloader.instances)}/{len(our_ids)}")

if len(downloader.instances) == 0:
    print("[ERREUR] Aucune instance filtrée trouvée. Vérifiez les IDs.")
else:
    if DOWNLOAD_POSES:
        downloader.include_cleaned_poses = True
        downloader._download_files(
            downloader._get_pose_origins(),
            title=f'Poses ({len(downloader.instances)} instances)'
        )

    if DOWNLOAD_POSES_RAW:
        downloader.include_raw_poses = True
        downloader._download_files(
            downloader._get_pose_origins(raw=True),
            title=f'Poses brutes ({len(downloader.instances)} instances)'
        )

    if DOWNLOAD_VIDEOS:
        downloader.include_videos = True
        downloader._download_files(
            downloader._get_video_origins(),
            title=f'Vidéos ({len(downloader.instances)} instances)'
        )

    print("\n[SUCCÈS] Dataset filtré téléchargé !")