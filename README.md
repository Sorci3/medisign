# MediSign Assist

Reconnaissance de signes LSF (Langue des Signes Française Belge) pour la communication d'urgence médicale avec des patients sourds.

## Description

MediSign Assist reconnaît 20 signes médicaux d'urgence en LSF à partir de coordonnées de pose squelettique. Deux approches sont implémentées :

- **TCN** (Temporal Convolutional Network) — approche recommandée, travaille sur les coordonnées brutes, adapté à la taille du dataset
- **I3D** (Inflated 3D Convolution Network) — approche de référence issue de l'article Fink et al. (2021), travaille sur des heatmaps 2D

### Signes reconnus (20 classes)

`SOUFFRIR` · `AIDER` · `FORT` · `MALADE` · `COEUR` · `TETE` · `MORT` · `DOS` · `VENTRE` · `FROID` · `JAMBE` · `JAMBES` · `RESPIRER` · `ACCIDENT` · `FAIBLE` · `ENCEINTE` · `DIABETE` · `BRAS` · `DOSSIER` · `EFFORT`

## Structure du projet

```
ProjetMajeur/
├── src/
│   ├── notebooks/
│   │   ├── 01_preprocessing.ipynb    # Prétraitement des données et génération des heatmaps (pour I3D)
│   │   ├── 02_i3d_training.ipynb     # Architecture I3D, entraînement, évaluation, sauvegarde
│   │   └── 03_tcn_training.ipynb     # Architecture TCN, entraînement, évaluation, sauvegarde
│   ├── dataset/
│   │   ├── instances.csv             # Métadonnées des instances
│   │   ├── poses/                    # Coordonnées de pose .npy (face, left_hand, right_hand, pose)
│   │   ├── videos/                   # Vidéos source .mp4
│   │   ├── metadata/                 # Splits train/test, mapping signe→index
│   │   └── preprocessed/             # Tenseurs heatmaps prétraités (générés par notebook 1, pour I3D)
│   ├── models/                       # Checkpoints des modèles entraînés
│   └── utils/
│       └── dataset_extraction.py     # Script de téléchargement du dataset
├── docs/
│   └── superpowers/plans/            # Plans d'implémentation
├── requirements.txt
└── fink2021.pdf                      # Article de référence (dataset LSFB + I3D)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

> **Note GPU :** PyTorch est installé en version CPU par défaut. Pour utiliser un GPU CUDA, réinstaller torch avec l'URL appropriée : https://pytorch.org/get-started/locally/

## Utilisation

### Approche recommandée — TCN (`src/notebooks/03_tcn_training.ipynb`)

Aucun prétraitement préalable nécessaire. Le notebook charge les poses directement depuis `src/dataset/poses/`.

1. **Chargement** des données brutes depuis les fichiers `.npy`
2. **Features** : coordonnées corps + mains concaténées → tenseur `(225, 32)` par instance
3. **Augmentation** probabiliste : flip horizontal/temporel, bruit spatial, time warp
4. **Architecture TCN** : projection 225→128 + 4 blocs temporels (dilations 1/2/4/8) + average pooling
5. **Entraînement** : AdamW + CosineAnnealingLR + early stopping (patience=15, max 80 époques)
6. **Évaluation** : accuracy, macro-F1, rapport par classe, matrice de confusion
7. **Sauvegarde** du modèle dans `src/models/tcn_medisign_final.pth`

### Approche de référence — I3D (`src/notebooks/01_preprocessing.ipynb` puis `02_i3d_training.ipynb`)

**Étape 1 — Prétraitement** (`01_preprocessing.ipynb`) :

1. Exploration des données et nettoyage (suppression des 4 instances T=0)
2. Génération de heatmaps Gaussiennes 64×64 pour corps, main gauche, main droite → tenseur `(3, 32, 64, 64)`
3. Augmentation ×4 : original + flip horizontal + flip temporel + bruit spatial
4. Sauvegarde dans `src/dataset/preprocessed/`

**Étape 2 — Entraînement** (`02_i3d_training.ipynb`) :

1. Architecture Inception-I3D (Carreira & Zisserman, 2017) pour entrée `(batch, 3, 32, 64, 64)`
2. Entraînement : Adam + ReduceLROnPlateau + early stopping (patience=10, max 50 époques)
3. Évaluation et sauvegarde dans `src/models/i3d_medisign_final.pth`

## Données

Le dataset provient de [LSFB ISOL v2](https://lsfb.info.unamur.be) (Belgian French Sign Language). Les fichiers de pose sont des tableaux NumPy de forme `(T, K, 3)` (float16) où :
- `T` = nombre de frames (variable, médiane ≈ 19)
- `K` = nombre de points clés (478 visage, 21 mains, 33 corps)
- `3` = coordonnées `(x, y, confiance)` normalisées dans `[0, 1]`

| Propriété | Valeur |
|---|---|
| Instances totales | 991 |
| Classes | 20 |
| Split entraînement | 607 instances |
| Split test | 380 instances |
| Classe majoritaire | FORT (486 instances) |
| Classe minoritaire | JAMBES (1 instance, uniquement dans le test) |

## Architectures

### TCN (recommandé)

```
Input: (batch, 225, T=32)   ← coordonnées brutes corps + mains
  │
  ├─ Conv1d projection 225 → 128
  ├─ TemporalBlock (dilation=1)
  ├─ TemporalBlock (dilation=2)
  ├─ TemporalBlock (dilation=4)
  ├─ TemporalBlock (dilation=8)
  │
  └─ AdaptiveAvgPool → Dropout(0.3) → Linear(128 → 20)

Paramètres : ~200k
```

### I3D (référence)

```
Input: (batch, 3, 32, 64, 64)   ← heatmaps Gaussiennes corps + mains
  │
  ├─ Stem: Conv3D 7×7×7 → MaxPool → Conv3D → Conv3D → MaxPool
  ├─ Mixed 3b, 3c → MaxPool
  ├─ Mixed 4b, 4c, 4d, 4e, 4f → MaxPool
  ├─ Mixed 5b, 5c
  │
  └─ AdaptiveAvgPool → Dropout(0.5) → Linear(1024 → 20)

Paramètres : ~12M
```

## Référence

Fink, M. et al. (2021). *LSFB-CONT and LSFB-ISOL: Two New Datasets for Vision-Based Sign Language Recognition.* — `fink2021.pdf`
