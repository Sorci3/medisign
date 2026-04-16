# MediSign Assist

Reconnaissance de signes LSF (Langue des Signes Française Belge) pour la communication d'urgence médicale avec des patients sourds. Ce projet implémente plusieurs architectures de pointe pour la reconnaissance de gestes isolés à partir de points clés squelettiques.

## 🚀 Vue d'ensemble

MediSign Assist reconnaît **20 signes médicaux d'urgence** en LSF à partir de coordonnées de pose extraites par MediaPipe. Le projet s'articule autour de trois approches majeures réparties sur différentes branches :

| Approche | Architecture | État | Points forts |
| :--- | :--- | :--- | :--- |
| **TCN (Main)** | Temporal Convolutional Network | **Recommandé** | Meilleur compromis vitesse/précision, pré-entraînement auto-supervisé. |
| **SPOTER** | Sign Language Recognition Transformer | **Alternative** | Basé sur le mécanisme d'attention, excellent pour les séquences longues. |
| **I3D** | Inflated 3D Convolution Network | **Référence** | Approche historique basée sur des heatmaps 2D (Fink et al., 2021). |

> **Signes reconnus :** `OUI`, `NON`, `APPELER`, `VITE`, `MANGER`, `COMPRENDRE`, `BOIRE`, `MALADE.VENTRE`, `CHAUD`, `DORMIR`, `RESPIRER`, `TOMBER`, `FROID`, `FAIBLE`, `ENCEINTE`, `HOPITAL`, `SOUFFRIR`, `PAS.SOUFFLER`, `FORT`, `MEDECIN`.

---

## 📂 Structure du Projet

```text
medisign-TCN-I3D/
├── src/                    # Branche Principale (TCN & I3D)
│   ├── notebooks/          # Pipelines d'entraînement (01-03)
│   ├── models/             # Checkpoints .pth et modèles MediaPipe
│   ├── utils/              # Scripts d'inférence et utilitaires
│   └── signs.py            # Registre central des classes
├── spoter/                 # Branche SPOTER (Dossier dédié)
│   ├── model.py            # Architecture Transformer
│   ├── train.py            # Entraînement & Fine-tuning
│   └── inference.py        # Inférence temps réel SPOTER
├── docs/                   # Documentation technique et plans
└── requirements.txt        # Dépendances du projet
```

---

## 🛠️ Installation

```bash
# Création de l'environnement
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installation des dépendances
pip install -r requirements.txt
```

---

## 🧠 Approche TCN (Branche Principale)

L'approche **TCN** est la plus aboutie de ce projet. Elle utilise une stratégie de **Transfer Learning** en deux phases :

1.  **Pré-entraînement Auto-supervisé** : Le modèle apprend à reconstruire des séquences de points clés masquées sur l'intégralité du dataset LSFB (~120 000 instances), sans labels.
2.  **Fine-tuning Supervisé** : Le modèle est affiné sur les 20 signes médicaux cibles avec des techniques de régularisation avancées (Label Smoothing, Mixup, Dropout Annealing).

### Inférence Temps Réel (TCN)
Le script d'inférence utilise le modèle **TCN pré-entraîné** pour une précision maximale.

```bash
python src/utils/inference.py
```

*   **Fonctionnement** : Détection automatique des mains via MediaPipe Holistic. Le buffer se remplit dès qu'un geste est détecté. Une prédiction est lancée dès que 32 frames sont accumulées, avec un vote majoritaire sur les 7 dernières prédictions pour stabiliser l'affichage.
*   **Contrôles** : `ESPACE` pour réinitialiser le buffer, `Q` pour quitter.

---

## 🤖 Approche SPOTER (Alternative Transformer)

Le dossier `spoter/` contient une implémentation basée sur les Transformers, adaptée du papier original *SPOTER*. Cette version utilise une normalisation géométrique spécifique (Bohacek) pour améliorer la robustesse aux variations de morphologie.

Pour plus de détails sur l'utilisation, l'entraînement et l'inférence de cette version, consultez le **[README dédié à SPOTER](./spoter/README.md)**.

---

## 📊 Référence I3D

L'approche **I3D** est conservée comme base de comparaison. Elle transforme les coordonnées squelettiques en heatmaps 2D (images) traitées par des convolutions 3D.
*   **Prétraitement** : `src/notebooks/01_i3d_preprocessing.ipynb`
*   **Entraînement** : `src/notebooks/02_i3d_training.ipynb`

---

## 📝 Maintenance & Données

Si vous souhaitez ré-extraire les poses ou mettre à jour le dataset :
1.  **Ré-extraction** : `python src/utils/extract_poses.py` (Aligne les données d'entraînement sur la version actuelle de MediaPipe).
2.  **Dataset** : Les données proviennent de [LSFB ISOL v2](https://lsfb.info.unamur.be).
