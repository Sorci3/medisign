# MediSign Assist

Reconnaissance de signes LSF (Langue des Signes Française Belge) pour la communication d'urgence médicale avec des patients sourds.

## Description

MediSign Assist utilise un modèle I3D (Inflated 3D Convolution Network) entraîné sur des coordonnées de pose squelettique pour reconnaître 20 signes médicaux d'urgence en LSF. L'entrée du modèle est constituée de cartes de chaleur 2D générées à partir des points clés du corps et des mains, permettant de travailler sans vidéo brute.

### Signes reconnus (20 classes)

`SOUFFRIR` · `AIDER` · `FORT` · `MALADE` · `COEUR` · `TETE` · `MORT` · `DOS` · `VENTRE` · `FROID` · `JAMBE` · `JAMBES` · `RESPIRER` · `ACCIDENT` · `FAIBLE` · `ENCEINTE` · `DIABETE` · `BRAS` · `DOSSIER` · `EFFORT`

## Structure du projet

```
ProjetMajeur/
├── src/
│   ├── notebooks/
│   │   ├── 01_preprocessing.ipynb    # Prétraitement des données et génération des heatmaps
│   │   └── 02_i3d_training.ipynb     # Architecture I3D, entraînement, évaluation, sauvegarde
│   ├── dataset/
│   │   ├── instances.csv             # Métadonnées des instances
│   │   ├── poses/                    # Coordonnées de pose .npy (face, left_hand, right_hand, pose)
│   │   ├── videos/                   # Vidéos source .mp4
│   │   ├── metadata/                 # Splits train/test, mapping signe→index
│   │   └── preprocessed/             # Tenseurs prétraités (générés par notebook 1)
│   ├── models/                       # Checkpoints du modèle (générés par notebook 2)
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

### Étape 1 — Prétraitement (`src/notebooks/01_preprocessing.ipynb`)

Ce notebook effectue dans l'ordre :

1. **Exploration** des données : distribution des classes, longueurs de séquences, visualisation du squelette
2. **Nettoyage** : suppression des 4 instances sans frames (T=0)
3. **Normalisation temporelle** : rééchantillonnage de chaque séquence à T=32 frames par interpolation linéaire
4. **Génération de heatmaps** : conversion des coordonnées (x, y, conf) en cartes de chaleur Gaussiennes 64×64 pour le corps, la main gauche et la main droite → tenseur `(3, 32, 64, 64)`
5. **Augmentation** (×4) : original + flip horizontal + flip temporel + bruit spatial
6. **Sauvegarde** des tenseurs dans `src/dataset/preprocessed/`

### Étape 2 — Entraînement (`src/notebooks/02_i3d_training.ipynb`)

1. **Chargement** des tenseurs prétraités
2. **Architecture I3D** : Inception-I3D (Carreira & Zisserman, 2017) adapté pour une entrée `(batch, 3, 32, 64, 64)`
3. **Entraînement** : Adam + ReduceLROnPlateau + early stopping (patience=10, max 50 époques) + poids de classe pour le déséquilibre
4. **Évaluation** : accuracy, macro-F1, rapport par classe, matrice de confusion
5. **Sauvegarde** du modèle dans `src/models/i3d_medisign_final.pth`

## Données

Le dataset provient de [LSFB ISOL v2](https://lsfb.info.unamur.be) (Belgian French Sign Language). Les fichiers de pose sont des tableaux NumPy de forme `(T, K, 3)` (float16) où :
- `T` = nombre de frames (variable, médiane ≈ 19)
- `K` = nombre de points clés (478 visage, 21 mains, 33 corps)
- `3` = coordonnées `(x, y, confiance)` normalisées dans `[0, 1]`

| Propriété | Valeur |
|---|---|
| Instances totales | 991 |
| Classes | 20 |
| Split entraînement | 611 instances |
| Split test | 380 instances |
| Classe majoritaire | FORT (486 instances) |
| Classe minoritaire | JAMBES (1 instance) |

## Architecture du modèle

```
Input: (batch, 3, 32, 64, 64)
  │
  ├─ Stem: Conv3D 7×7×7 stride 2 → MaxPool → Conv3D 1×1×1 → Conv3D 3×3×3 → MaxPool
  │
  ├─ Mixed 3b, 3c → MaxPool
  ├─ Mixed 4b, 4c, 4d, 4e, 4f → MaxPool
  ├─ Mixed 5b, 5c
  │
  └─ AdaptiveAvgPool → Dropout(0.5) → Linear(1024 → 20)
```

Chaque bloc `InceptionModule` combine 4 branches : `1×1×1`, `1×1×1→3×3×3`, `1×1×1→3×3×3`, `MaxPool→1×1×1`.

## Référence

Fink, M. et al. (2021). *LSFB-CONT and LSFB-ISOL: Two New Datasets for Vision-Based Sign Language Recognition.* — `fink2021.pdf`
