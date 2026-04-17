# pretrain.py — Pré-entraînement SPOTER sur le dataset LSFB complet (452 classes)
#
# But : donner à SPOTER une représentation générale de la langue des signes
# avant de le spécialiser sur nos 20 signes médicaux.
#
# Workflow :
#   1. python prepare_data.py --step download             # télécharge les poses
#   2. python prepare_data.py --step pretrain-landmarks   # normalise + sauvegarde
#   3. python pretrain.py                                 # pré-entraîne SPOTER
#   4. python train.py --pretrained models/spoter/pretrained.pt
#

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import SPOTER

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Config ───────────────────────────────────────────────────────────────────
with open(os.path.join(_HERE, "config.json")) as f:
    CFG = json.load(f)

DATASET         = os.path.realpath(os.path.join(_HERE, CFG["dataset_path"]))
LANDMARK_DIR    = os.path.join(DATASET, "pretrain_landmarks")
INSTANCES_CSV   = os.path.join(DATASET, "instances.csv")
MODELS_DIR     = os.path.join(_HERE, "models", "spoter")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_SIZE = 225   # pas de face pour le pré-entraînement
MAX_LEN      = 80    # taille maximale de la séquence
MIN_INSTANCES = 50   # filtre les signes avec trop peu d'exemples
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset (lazy loading depuis disque) ────────────────────────────────────

class SignDataset(Dataset):
    """
    Charge chaque séquence à la volée depuis pretrain_landmarks/ dans __getitem__,.
    Les fichiers sont déjà normalisés.
    """
    def __init__(self, paths, labels):
        self.paths  = paths   # liste de chemins vers les .npy
        self.labels = labels

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        seq = np.load(self.paths[i])          # charge depuis disque
        T   = min(len(seq), MAX_LEN)
        pad = np.zeros((MAX_LEN, FEATURE_SIZE), dtype=np.float32)
        pad[MAX_LEN - T:] = seq[:T]           # zero-pad à gauche
        return (torch.tensor(pad, dtype=torch.float32),
                torch.tensor(int(self.labels[i]), dtype=torch.long))


def class_weights(y, n_classes):
    counts  = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(counts + 1e-6)
    return weights / weights.sum() * n_classes


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = correct = total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(y_b)
        correct  += (out.argmax(1) == y_b).sum().item()
        total    += len(y_b)
    return loss_sum / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            out  = model(X_b)
            loss_sum += criterion(out, y_b).item() * len(y_b)
            correct  += (out.argmax(1) == y_b).sum().item()
            total    += len(y_b)
    return loss_sum / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(LANDMARK_DIR):
        print(f"pretrain_landmarks/ introuvable : {LANDMARK_DIR}")
        print("Lancez d'abord : python prepare_data.py --step pretrain-landmarks")
        return

    epochs    = CFG["pretrain"]["epochs"]
    batch_sz  = CFG["pretrain"]["batch_size"]
    lr        = CFG["pretrain"]["learning_rate"]
    patience  = CFG["pretrain"]["patience"]

    # ── Sélection des signes avec assez d'instances ───────────────────────────
    index_csv     = os.path.join(DATASET, "index.csv")
    medical_signs = set(pd.read_csv(index_csv)["sign"]) if os.path.exists(index_csv) else set()

    df     = pd.read_csv(INSTANCES_CSV)
    df     = df[~df["sign"].isin(medical_signs)].copy()   # exclut les signes du fine-tuning
    counts = df["sign"].value_counts()
    df     = df[df["sign"].isin(counts[counts >= MIN_INSTANCES].index)].copy()

    # Garde uniquement les instances dont le fichier existe
    df["path"] = df["id"].apply(lambda x: os.path.join(LANDMARK_DIR, x + ".npy"))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    n_classes = df["sign"].nunique()
    print(f"device : {DEVICE}")
    print(f"{len(df)} instances, {n_classes} classes (seuil >= {MIN_INSTANCES})")

    le = LabelEncoder()
    le.fit(sorted(df["sign"].unique()))
    df["label"] = le.transform(df["sign"])
    y_int = df["label"].values

    # ── Split train / val ─────────────────────────────────────────────────────
    tr_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.10, stratify=y_int, random_state=42)

    paths_tr  = df["path"].iloc[list(tr_idx)].tolist()
    paths_val = df["path"].iloc[list(val_idx)].tolist()
    y_tr      = y_int[list(tr_idx)]
    y_val     = y_int[list(val_idx)]
    print(f"train: {len(paths_tr)}, val: {len(paths_val)}")

    cw       = class_weights(y_tr, n_classes)
    sample_w = torch.tensor([cw[lbl] for lbl in y_tr], dtype=torch.float32)
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(SignDataset(paths_tr, y_tr),   batch_size=batch_sz,
                              sampler=sampler, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(SignDataset(paths_val, y_val), batch_size=batch_sz,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = SPOTER(
        num_classes        = n_classes,
        feature_size       = FEATURE_SIZE,
        hidden_dim         = CFG["model"]["hidden_dim"],
        nhead              = CFG["model"]["nhead"],
        num_encoder_layers = CFG["model"]["num_encoder_layers"],
        num_decoder_layers = CFG["model"]["num_decoder_layers"],
        dim_feedforward    = CFG["model"]["dim_feedforward"],
        dropout            = CFG["pretrain"]["dropout"],
    ).to(DEVICE)
    print(f"SPOTER — {sum(p.numel() for p in model.parameters()):,} parametres — {n_classes} classes")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE),
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    ckpt_path     = os.path.join(MODELS_DIR, "pretrained.pt")
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = []

    for epoch in range(1, epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer)
        vl, va = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va})
        print(f"epoch {epoch:03d}/{epochs}  "
              f"train loss={tl:.4f} acc={ta:.4f}  val loss={vl:.4f} acc={va:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl; patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping a l'epoch {epoch}"); break

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    meta = {
        "n_classes_pretrain": n_classes, "feature_size": FEATURE_SIZE,
        "hidden_dim": CFG["model"]["hidden_dim"], "max_len": MAX_LEN,
        "best_val_loss": best_val_loss,
    }
    with open(os.path.join(MODELS_DIR, "pretrained_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPre-entrainement termine. Poids : {ckpt_path}")
    print(f"Lancez le fine-tuning :")
    print(f"  python train.py --pretrained {ckpt_path}")


if __name__ == "__main__":
    main()
