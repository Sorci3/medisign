# MediSign — SPOTER (Sign Language Transformer)

Cette branche contient l'implémentation de l'architecture **SPOTER** (Sign Language Recognition Transformer) adaptée pour la reconnaissance des 20 signes médicaux du projet MediSign.

## 🧐 Pourquoi SPOTER ?

Contrairement aux approches convolutives (TCN/I3D), SPOTER repose entièrement sur le mécanisme d'**Attention**. Il est particulièrement efficace pour capturer les relations temporelles complexes dans les gestes de la langue des signes et utilise une normalisation géométrique robuste.

### Caractéristiques clés :
- **Architecture** : Transformer Encoder-Decoder simplifié (sans auto-attention dans le décodeur).
- **Normalisation Bohacek** : Recalage du corps par rapport aux épaules et normalisation locale des mains.
- **2-Phase Training** : Support du pré-entraînement sur le dataset complet LSFB suivi d'un fine-tuning ciblé.

---

## 🚀 Utilisation

### 1. Préparation des données
Le script `prepare_data.py` automatise tout le pipeline :
```bash
# Téléchargement, indexation et fusion des landmarks
python prepare_data.py
```
*Note : Les données sont stockées dans le dossier défini dans `config.json`.*

### 2. Entraînement
Vous pouvez entraîner le modèle de deux manières :

**A. À partir de zéro (From Scratch) :**
```bash
python train.py
```

**B. Fine-tuning (Recommandé) :**
Si vous avez un modèle pré-entraîné (`models/spoter/pretrained.pt`) :
```bash
python train.py --pretrained models/spoter/pretrained.pt --freeze-epochs 10
```
*La phase 1 entraîne uniquement la tête de classification (10 époques), la phase 2 dégèle tout le réseau.*

### 3. Inférence Temps Réel
Lancez la reconnaissance via webcam avec le modèle SPOTER :
```bash
python inference.py
```
- **Cycle** : Collecte 60 frames (~2s), prédit, puis réinitialise.
- **Affichage** : Top-5 des candidats avec barres de confiance.
- **Touches** : `c` pour reset manuel, `q` pour quitter.

---

## 🏗️ Architecture Technique

Le modèle est configuré via `config.json` :
- **Hidden Dim** : 64
- **Heads** : 4
- **Layers** : 6 Encodeurs / 6 Décodeurs
- **Features** : 225 (33 corps + 21 main gauche + 21 main droite, x/y/z)

Les résultats (courbes d'apprentissage, matrice de confusion) sont automatiquement sauvegardés dans le dossier `results/` après chaque entraînement.

---

## 📚 Références
Adapté de : *Boháček, M., & Hrúz, M. (2022). SPOTER: Sign Pose-Based Transformer for Isolated Sign Language Recognition.*
