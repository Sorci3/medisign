# train.py — Entraînement SPOTER sur les 20 signes médicaux
#
# Modes :
#   From scratch  : python train.py
#   Fine-tuning   : python train.py --pretrained models/spoter/pretrained.pt
#
# Avec --pretrained, le 2-phase training s'active :
#   Phase 1 (--freeze-epochs, défaut 10) : seule la tête de classification est entraînée
#   Phase 2 (epochs restantes)           : tout le réseau, LR divisé par 10

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import SPOTER
from utils import normalize_sequence, augment_class

# ─── Config ───────────────────────────────────────────────────────────────────
with open(os.path.join(_HERE, "config.json")) as f:
    CFG = json.load(f)

DATASET      = os.path.realpath(os.path.join(_HERE, CFG["dataset_path"]))
LANDMARK_DIR = os.path.join(DATASET, "landmarks")
INDEX_PATH   = os.path.join(DATASET, "index.csv")
MODELS_DIR   = os.path.join(_HERE, "models", "spoter")
RESULTS_DIR  = os.path.join(_HERE, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_SIZE = 225
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {DEVICE}")


# ─── Dataset PyTorch ──────────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def load_data(df):
    """Charge les fichiers .npy depuis landmarks/ et retourne les séquences brutes."""
    X, y, miss = [], [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="chargement"):
        path = os.path.join(LANDMARK_DIR, row["id"] + ".npy")
        if not os.path.exists(path):
            miss += 1; continue
        X.append(np.load(path))
        y.append(row["sign"])
    if miss:
        print(f"attention : {miss} fichiers manquants")
    return X, y


def pad_sequences(sequences, max_len):
    """Zero-padding à gauche → (N, max_len, FEATURE_SIZE)."""
    result = np.zeros((len(sequences), max_len, FEATURE_SIZE), dtype=np.float32)
    for i, seq in enumerate(sequences):
        l = min(len(seq), max_len)
        result[i, max_len - l:] = seq[:l]
    return result


def class_weights(y, n_classes):
    """
    Pondération 1/sqrt(n) : atténue le déséquilibre sans l'inverser.
    1/n écrasait OUI (2300 samples) à 0 ; sqrt conserve sa présence.
    """
    counts  = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(counts + 1e-6)
    return weights / weights.sum() * n_classes


# ─── Boucles d'entraînement ───────────────────────────────────────────────────

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
            loss = criterion(out, y_b)
            loss_sum += loss.item() * len(y_b)
            correct  += (out.argmax(1) == y_b).sum().item()
            total    += len(y_b)
    return loss_sum / total, correct / total


# ─── Sauvegarde des courbes et de la matrice de confusion ────────────────────

def save_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"]   for h in history], label="val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"]   for h in history], label="val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_confusion(y_true, y_pred, classes, path, acc):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
    ax.set_title(f"Confusion SPOTER — {acc*100:.1f}%")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=None,
                        help="Chemin vers pretrained.pt pour activer le fine-tuning 2-phase")
    parser.add_argument("--freeze-epochs", type=int, default=10,
                        help="Nombre d'epochs phase 1 (tête seule, défaut: 10)")
    args = parser.parse_args()

    epochs    = CFG["epochs"]
    batch_sz  = CFG["batch_size"]
    lr        = CFG["learning_rate"]
    patience  = CFG["patience"]
    min_samp  = CFG["min_samples_after_augment"]

    # ── Chargement + normalisation ────────────────────────────────────────────
    df = pd.read_csv(INDEX_PATH)
    X_raw, y_raw = load_data(df)
    print("Normalisation Bohacek...")
    X_raw = [np.nan_to_num(normalize_sequence(s)) for s in tqdm(X_raw)]

    le = LabelEncoder()
    le.fit(sorted(df["sign"].unique()))
    y_int     = le.transform(y_raw)
    n_classes = len(le.classes_)
    print(f"\n{len(X_raw)} sequences, {n_classes} classes : {le.classes_.tolist()}")

    # ── Split train / val / test ──────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_raw, y_int, test_size=CFG["test_size"], stratify=y_int, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=CFG["val_size"] / (1 - CFG["test_size"]),
        stratify=y_tmp, random_state=42)
    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── Augmentation des classes sous-représentées ────────────────────────────
    cs = defaultdict(list)
    for seq, lbl in zip(X_train, y_train):
        cs[lbl].append(seq)
    aug_X, aug_y = list(X_train), list(y_train)
    for lbl, samp in cs.items():
        if len(samp) < min_samp:
            new = augment_class(samp, min_samp)
            aug_X.extend(new)
            aug_y.extend([lbl] * len(new))
            print(f"  aug {le.classes_[lbl]} : {len(samp)} -> {len(samp)+len(new)}")
    perm  = np.random.permutation(len(aug_X))
    aug_X = [aug_X[i] for i in perm]
    aug_y = np.array([aug_y[i] for i in perm])
    print(f"train apres augmentation : {len(aug_X)}")

    # ── Padding + DataLoaders ─────────────────────────────────────────────────
    max_len      = max(len(s) for s in aug_X + X_val + X_test)
    X_train_pad  = pad_sequences(aug_X, max_len)
    X_val_pad    = pad_sequences(X_val, max_len)
    X_test_pad   = pad_sequences(X_test, max_len)

    cw       = class_weights(aug_y, n_classes)
    sample_w = torch.tensor([cw[lbl] for lbl in aug_y], dtype=torch.float32)
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(SignDataset(X_train_pad, aug_y), batch_size=batch_sz, sampler=sampler)
    val_loader   = DataLoader(SignDataset(X_val_pad, y_val),   batch_size=batch_sz)
    test_loader  = DataLoader(SignDataset(X_test_pad, y_test), batch_size=batch_sz)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = SPOTER(
        num_classes        = n_classes,
        feature_size       = FEATURE_SIZE,
        hidden_dim         = CFG["hidden_dim"],
        nhead              = CFG["nhead"],
        num_encoder_layers = CFG["num_encoder_layers"],
        num_decoder_layers = CFG["num_decoder_layers"],
        dim_feedforward    = CFG["dim_feedforward"],
        dropout            = CFG["dropout"],
    ).to(DEVICE)
    print(f"\nSPOTER — {sum(p.numel() for p in model.parameters()):,} parametres")

    # ── Chargement pré-entraîné (optionnel) ───────────────────────────────────
    use_pretrained = args.pretrained and os.path.exists(args.pretrained)
    if use_pretrained:
        state = torch.load(args.pretrained, map_location=DEVICE)
        compatible = {k: v for k, v in state.items() if not k.startswith("linear_class")}
        model.load_state_dict(compatible, strict=False)
        print(f"Poids pre-entraines charges depuis {args.pretrained}")
        # Phase 1 : gèle tout sauf la tête
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("linear_class")
        print(f"Phase 1 : {args.freeze_epochs} epochs — tete seule")
    else:
        print("Entrainement from scratch")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(aug_y, n_classes), dtype=torch.float32).to(DEVICE),
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    ckpt_path      = os.path.join(MODELS_DIR, "best_model.pt")
    best_val_loss  = float("inf")
    patience_ctr   = 0
    history        = []
    phase2_started = not use_pretrained

    for epoch in range(1, epochs + 1):
        # Passage phase 1 → phase 2
        if use_pretrained and not phase2_started and epoch > args.freeze_epochs:
            print(f"\nPhase 2 : degel de tout le reseau (LR = {lr * 0.1:.2e})")
            for p in model.parameters():
                p.requires_grad = True
            optimizer    = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            best_val_loss = float("inf")
            patience_ctr  = 0
            phase2_started = True

        tl, ta = train_epoch(model, train_loader, criterion, optimizer)
        vl, va = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va})
        print(f"[{epoch:03d}/{epochs}]  train loss={tl:.4f} acc={ta:.3f}  val loss={vl:.4f} acc={va:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl; patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> checkpoint sauvegarde")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping a l'epoch {epoch}"); break

    # ── Évaluation finale ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"\ntest accuracy : {test_acc*100:.1f}%")

    all_preds = []
    model.eval()
    with torch.no_grad():
        for X_b, _ in test_loader:
            all_preds.extend(model(X_b.to(DEVICE)).argmax(1).cpu().numpy())

    report = classification_report(y_test, all_preds, target_names=le.classes_)
    print(report)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    meta = {
        "max_len": max_len, "n_classes": n_classes, "feature_size": FEATURE_SIZE,
        "hidden_dim": CFG["hidden_dim"], "nhead": CFG["nhead"],
        "num_encoder_layers": CFG["num_encoder_layers"],
        "num_decoder_layers": CFG["num_decoder_layers"],
        "dim_feedforward": CFG["dim_feedforward"],
        "dropout": CFG["dropout"],
        "test_accuracy": test_acc,
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(MODELS_DIR, "classes.json"), "w") as f:
        json.dump(le.classes_.tolist(), f)

    run_dir = os.path.join(RESULTS_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(f"test accuracy : {test_acc:.4f}\n\n{report}")
    save_curves(history, os.path.join(run_dir, "curves.png"))
    save_confusion(y_test, all_preds, le.classes_,
                   os.path.join(run_dir, "confusion.png"), test_acc)
    print(f"\nResultats sauvegardes dans {run_dir}")


if __name__ == "__main__":
    main()
