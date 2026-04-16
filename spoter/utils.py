# utils.py — Normalisation et augmentation de données
#
# Deux responsabilités :
#   1. Normalisation de Bohacek : rend les landmarks invariants à la position/taille
#      du signeur dans le cadre. Appliquée à TOUTES les séquences (train, val, test,
#      inférence).
#   2. Augmentations : génère des variations synthétiques pour les classes sous-représentées.
#      Appliquées uniquement sur le train set.

import numpy as np

# ─── Structure des features MediaPipe ─────────────────────────────────────────
# 225 features (sans visage) :
#   indices   0 -  98 : 33 joints corps  × 3 (x, y, z)
#   indices  99 - 161 : 21 joints main gauche × 3
#   indices 162 - 224 : 21 joints main droite × 3

N_BODY           = 33
N_HAND           = 21
LEFT_HAND_START  = N_BODY * 3                        # 99
RIGHT_HAND_START = LEFT_HAND_START + N_HAND * 3      # 162
RIGHT_HAND_END   = RIGHT_HAND_START + N_HAND * 3     # 225

# Indices des joints clés pour la normalisation corps
_LS_X, _LS_Y = 33, 34   # leftShoulder  (joint 11 × 3)
_RS_X, _RS_Y = 36, 37   # rightShoulder (joint 12 × 3)
_LE_Y        = 7         # leftEye y     (joint  2 × 3 + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISATION DE BOHACEK
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_body(seq: np.ndarray) -> np.ndarray:
    """
    Normalise les 33 joints du corps dans une boîte centrée sur le cou,
    de côté 6× la distance inter-épaules.

    Résultat : les coordonnées du corps sont indépendantes de la position
    et de la taille du signeur dans le cadre.
    """
    out = seq.copy()

    ls_x, ls_y = seq[:, _LS_X], seq[:, _LS_Y]
    rs_x, rs_y = seq[:, _RS_X], seq[:, _RS_Y]
    le_y       = seq[:, _LE_Y]

    neck_x      = (ls_x + rs_x) * 0.5
    head_metric = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)

    sp_x = neck_x - 3.0 * head_metric
    sp_y = le_y   + head_metric
    ep_x = neck_x + 3.0 * head_metric
    ep_y = sp_y   - 6.0 * head_metric

    bbox_w = ep_x - sp_x
    bbox_h = sp_y - ep_y
    valid  = (bbox_w > 1e-6) & (bbox_h > 1e-6) & (ls_x != 0) & (rs_x != 0)

    for j in range(N_BODY):
        xi, yi = j * 3, j * 3 + 1
        out[:, xi] = np.where(valid, (seq[:, xi] - sp_x) / bbox_w, seq[:, xi])
        out[:, yi] = np.where(valid, (seq[:, yi] - ep_y) / bbox_h, seq[:, yi])

    return out


def normalize_hands(seq: np.ndarray) -> np.ndarray:
    """
    Normalise chaque main dans sa propre boîte englobante, frame par frame.

    Résultat : la position absolue de la main dans le cadre est effacée ;
    seule la configuration des doigts (forme) est conservée.
    """
    out = seq.copy()

    for hand_start in (LEFT_HAND_START, RIGHT_HAND_START):
        x_idx = np.arange(hand_start,     hand_start + N_HAND * 3, 3)
        y_idx = np.arange(hand_start + 1, hand_start + N_HAND * 3, 3)

        for t in range(len(seq)):
            x_vals, y_vals = seq[t, x_idx], seq[t, y_idx]
            mask = x_vals != 0
            if mask.sum() < 3:
                continue

            vx, vy = x_vals[mask], y_vals[mask]
            w, h   = vx.max() - vx.min(), vy.max() - vy.min()

            if w > h:
                dx, dy = 0.1 * w, 0.1 * w + (w - h) * 0.5
            else:
                dy, dx = 0.1 * h, 0.1 * h + (h - w) * 0.5

            sp_x, ep_x = vx.min() - dx, vx.max() + dx
            sp_y, ep_y = vy.min() - dy, vy.max() + dy
            bw, bh     = ep_x - sp_x, ep_y - sp_y

            if bw < 1e-6 or bh < 1e-6:
                continue

            for j in range(N_HAND):
                if seq[t, x_idx[j]] == 0:
                    continue
                out[t, x_idx[j]] = (seq[t, x_idx[j]] - sp_x) / bw
                out[t, y_idx[j]] = (seq[t, y_idx[j]] - sp_y) / bh

    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Point d'entrée : normalise corps puis mains. Input/output : (T, F)."""
    return normalize_hands(normalize_body(seq))


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def flip_horizontal(seq):
    """Miroir horizontal + échange main gauche/droite."""
    s = seq.copy()
    for start, n in [(0, N_BODY), (LEFT_HAND_START, N_HAND), (RIGHT_HAND_START, N_HAND)]:
        for j in range(n):
            s[:, start + j * 3] = 1.0 - s[:, start + j * 3]
    left  = s[:, LEFT_HAND_START:RIGHT_HAND_START].copy()
    right = s[:, RIGHT_HAND_START:RIGHT_HAND_END].copy()
    s[:, LEFT_HAND_START:RIGHT_HAND_START] = right
    s[:, RIGHT_HAND_START:RIGHT_HAND_END]  = left
    return s


def add_noise(seq, std=0.005):
    """Bruit gaussien sur tous les landmarks."""
    return (seq + np.random.normal(0, std, seq.shape)).astype(np.float32)


def scale(seq, low=0.9, high=1.1):
    """Mise à l'échelle globale centrée en 0.5 — simule la distance signeur/caméra."""
    return ((seq - 0.5) * np.random.uniform(low, high) + 0.5).astype(np.float32)


def time_warp(seq, low=0.8, high=1.2):
    """Rééchantillonnage temporel — simule des variations de vitesse du signe."""
    T, F  = seq.shape
    new_T = max(2, int(T * np.random.uniform(low, high)))
    out   = np.empty((new_T, F), dtype=np.float32)
    for i in range(new_T):
        idx  = i * (T - 1) / (new_T - 1)
        lo   = int(idx)
        hi   = min(lo + 1, T - 1)
        a    = idx - lo
        out[i] = seq[lo] * (1 - a) + seq[hi] * a
    return out


def temporal_permutation(seq):
    """
    Permute les frames périphériques (début N//3, fin N//4), préserve le centre.
    Source : SL-SLR (arXiv 2509.05188).
    """
    seq = seq.copy()
    T   = len(seq)
    n1, n2 = T // 3, T // 4
    if n1 > 1:
        seq[:n1] = seq[np.random.permutation(n1)]
    if n2 > 1:
        seq[T - n2:] = seq[T - n2:][np.random.permutation(n2)]
    return seq


def hand_local_noise(seq, std=0.02):
    """Bruit plus fort sur les mains — inspire de Fink et al. IDA 2025."""
    out = seq.copy()
    out[:, LEFT_HAND_START:RIGHT_HAND_END] += np.random.normal(
        0, std, (len(seq), RIGHT_HAND_END - LEFT_HAND_START))
    return out.astype(np.float32)


def hand_scale(seq, low=0.85, high=1.15):
    """Mise à l'échelle locale des mains autour du poignet — inspire de Fink et al. IDA 2025."""
    factor = np.random.uniform(low, high)
    out    = seq.copy()
    T      = len(seq)
    for hand_start, hand_end in [(LEFT_HAND_START, RIGHT_HAND_START),
                                  (RIGHT_HAND_START, RIGHT_HAND_END)]:
        wrist = out[:, hand_start:hand_start + 3]
        hand  = out[:, hand_start:hand_end].reshape(T, N_HAND, 3)
        hand  = wrist[:, np.newaxis, :] + factor * (hand - wrist[:, np.newaxis, :])
        out[:, hand_start:hand_end] = hand.reshape(T, N_HAND * 3)
    return out.astype(np.float32)


def temporal_shift(seq, max_shift=0.1):
    """Supprime jusqu'à 10% des frames au début ou à la fin."""
    T     = len(seq)
    shift = np.random.randint(0, max(1, int(T * max_shift)) + 1)
    if shift == 0:
        return seq
    return seq[shift:] if np.random.rand() < 0.5 else seq[:-shift]


# Liste utilisée par train.py
AUGMENTATIONS = [
    flip_horizontal,
    add_noise,
    scale,
    time_warp,
    temporal_permutation,
    hand_local_noise,
    hand_scale,
    temporal_shift,
]


def augment_class(samples, target):
    """Sur-échantillonne une classe jusqu'à `target` exemples en appliquant des augmentations aléatoires."""
    new = []
    while len(samples) + len(new) < target:
        src     = samples[np.random.randint(len(samples))]
        aug_fn  = AUGMENTATIONS[np.random.randint(len(AUGMENTATIONS))]
        new.append(aug_fn(src))
    return new
