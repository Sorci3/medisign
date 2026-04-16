# Télécharge toutes les poses LSFB disponibles (dataset complet).
# - Pas de vidéos
# - Pas de poses face sauf si MEDISIGN_USE_FACE=1
# - Reprend là où il s'est arrêté si annulé
#
# Usage :
#   python data/download_pretrain_data.py
#   MEDISIGN_USE_FACE=1 python data/download_pretrain_data.py   # avec face

import os
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from lsfb_dataset import Downloader

_HERE    = os.path.dirname(os.path.abspath(__file__))
_CURRENT = os.path.join(_HERE, "..")
with open(os.path.join(_HERE, "config.json")) as _f:
    _cfg = json.load(_f)
DESTINATION = os.path.realpath(os.path.join(_HERE, _cfg["dataset_path"]))
INSTANCES_CSV = os.path.join(DESTINATION, "instances.csv")
POSES_DIR     = os.path.join(DESTINATION, "poses", "pose")
WORKERS       = 32


def make_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=WORKERS, pool_maxsize=WORKERS)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

_session: requests.Session = None

def download_one(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        return True
    try:
        r = _session.get(url, timeout=30)
        if r.status_code == 200:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception:
        return False


def main():
    # Créer le dossier de destination si nécessaire
    os.makedirs(DESTINATION, exist_ok=True)

    # Télécharger d'abord les métadonnées (instances.csv) si absent
    print("Récupération des métadonnées LSFB...")
    BASE_URL   = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"
    use_face   = os.environ.get("MEDISIGN_USE_FACE", "0") == "1"
    downloader = Downloader(
        dataset="isol", destination=DESTINATION, splits=["all"],
        include_videos=False, include_cleaned_poses=False,
        include_raw_poses=False, skip_existing_files=True,
    )
    downloader.download()

    df = pd.read_csv(INSTANCES_CSV)
    all_ids = set(df["id"].tolist())

    already = ({os.path.splitext(f)[0] for f in os.listdir(POSES_DIR) if f.endswith(".npy")}
               if os.path.exists(POSES_DIR) else set())
    to_download = all_ids - already

    print(f"Dataset complet   : {len(all_ids):,} instances, {df['sign'].nunique():,} signes")
    print(f"Déjà téléchargées : {len(already):,}")
    print(f"À télécharger     : {len(to_download):,}")

    if not to_download:
        print("Tout est déjà téléchargé.")
        return

    # Filtrer les instances restantes et récupérer les chemins de poses
    downloader.instances = [i for i in downloader.instances if i in to_download]
    downloader.include_cleaned_poses = True
    origins = downloader._get_pose_origins()

    tasks = []
    for rel_path in origins:
        if "/face/" in rel_path and not use_face:
            continue
        url  = f"{BASE_URL}/{rel_path}"
        dest = os.path.join(DESTINATION, rel_path.replace("/", os.sep))
        if not os.path.exists(dest):
            tasks.append((url, dest))

    n_files = len(tasks)
    est_gb  = n_files * 9 / 1024 / 1024
    print(f"Fichiers à télécharger : {n_files:,}  (~{est_gb:.1f} GB)")
    print(f"Threads                : {WORKERS}")

    confirm = input("Continuer ? [o/N] ").strip().lower()
    if confirm not in ("o", "oui", "y", "yes"):
        print("Annulé.")
        return

    global _session
    _session = make_session()

    errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_one, url, dest): dest for url, dest in tasks}
        with tqdm(total=n_files, unit="fichiers") as pbar:
            for future in as_completed(futures):
                if not future.result():
                    errors += 1
                pbar.update(1)
                if errors:
                    pbar.set_postfix(erreurs=errors)

    print(f"\nTerminé — {n_files - errors}/{n_files} fichiers OK"
          + (f", {errors} erreurs" if errors else " ✓"))
    print("\nLancez ensuite :")
    print("  python data/make_pretrain_index.py")


if __name__ == "__main__":
    main()
