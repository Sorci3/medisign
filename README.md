# MediSign — SPOTER (Sign Language Transformer)

Cette branche contient l'implémentation de l'architecture **SPOTER** (Sign Language Recognition Transformer) adaptée pour la reconnaissance des 20 signes médicaux du projet MediSign.

## 🧐 Pourquoi SPOTER ?

Contrairement aux approches convolutives (TCN/I3D), SPOTER repose entièrement sur le mécanisme d'**Attention**. Il est particulièrement efficace pour capturer les relations temporelles complexes dans les gestes de la langue des signes et utilise une normalisation géométrique robuste.

### Caractéristiques clés :
- **Architecture** : Transformer Encoder-Decoder simplifié (sans auto-attention dans le décodeur).
- **Normalisation Bohacek** : Recalage du corps par rapport aux épaules et normalisation locale des mains.
- **Entraînement en 2 phases** : Pré-entraînement sur l'intégralité du dataset LSFB pour apprendre la structure des signes, suivi d'un fine-tuning ciblé sur notre dictionnaire médical.

---

## 🚀 Utilisation

### 1. Préparation des données
Le script `prepare_data.py` automatise tout le pipeline de données :
```bash
# Téléchargement, indexation et fusion des landmarks
python prepare_data.py
```
*Note : Les données sont stockées dans le dossier défini dans `config.json`.*

### 2. Pipeline d'Entraînement
Le modèle suit un cycle d'apprentissage complet pour maximiser sa précision :

**Phase A : Pré-entraînement (Généraliste)**
On entraîne d'abord le modèle sur tout le dataset LSFB pour qu'il apprenne à reconnaître une grande variété de formes et de mouvements.
```bash
python pretrain.py
```

**Phase B : Fine-tuning (Spécialisé)**
On affine ensuite le modèle sur les 20 signes médicaux cibles. Cette étape utilise les poids pré-entraînés pour accélérer la convergence.
```bash
python train.py --pretrained models/spoter/pretrained.pt --freeze-epochs 10
```
*Le script gèle le backbone pendant 10 époques pour stabiliser la tête de classification avant de libérer l'ensemble du réseau.*

### 3. Inférence Temps Réel
Lancez la reconnaissance via webcam avec le modèle SPOTER final :
```bash
python inference.py
```
- **Cycle** : Collecte 60 frames (~2s), prédit, puis réinitialise automatiquement.
- **Affichage** : Top-5 des candidats avec barres de confiance en temps réel.
- **Touches** : `c` pour reset manuel du buffer, `q` pour quitter.

---

## 🏗️ Architecture Technique

Le modèle est configuré via `config.json` :
- **Hidden Dim** : 64
- **Heads** : 4
- **Layers** : 6 Encodeurs / 6 Décodeurs
- **Features** : 225 (33 corps + 21 main gauche + 21 main droite, x/y/z)

Les résultats (courbes d'apprentissage, matrice de confusion, rapports de performance) sont automatiquement sauvegardés dans le dossier `results/` après chaque session.

---

## 📚 Références
Adapté de : *Boháček, M., & Hrúz, M. (2022). SPOTER: Sign Pose-Based Transformer for Isolated Sign Language Recognition.*
