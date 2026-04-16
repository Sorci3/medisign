# MediSign Assist

Reconnaissance de signes LSFB (Langue des Signes Française Belge) pour la communication d'urgence médicale avec des patients sourds. Ce projet a été réalisé dans le cadre de nos études à l'**ESAIP**, au sein de la spécialisation **Intelligence Artificielle**.

### 👥 Équipe Projet
*   **Sorci3**
*   **Mathis Herbreteau**
*   **Axel Bonneau**
*   **Mathis Marsault**
*   **Louis Maillet**

---

## 🚀 Notre Approche : TCN-I3D (Branche Main)

La branche principale de ce projet, intitulée **TCN-I3D**, regroupe nos travaux majeurs sur la reconnaissance de gestes isolés à partir de points clés squelettiques. Elle propose deux architectures complémentaires pour traiter les 20 signes médicaux cibles.

### 🧠 Approche TCN (Temporal Convolutional Network)
L'approche **TCN** est notre méthode recommandée pour sa rapidité et sa précision. Elle utilise une stratégie de **Transfer Learning** avancée :
1.  **Pré-entraînement Auto-supervisé** : Le modèle apprend les structures de mouvement sur l'intégralité du dataset LSFB (~120 000 séquences) par masquage temporel, sans labels.
2.  **Fine-tuning Supervisé** : Affinement sur les signes médicaux avec régularisation (Label Smoothing, Mixup) pour une robustesse accrue.

### 📊 Approche I3D (Inflated 3D ConvNet)
En complément, nous avons implémenté l'approche **I3D**, qui sert de référence historique. Cette méthode transforme les coordonnées squelettiques en **heatmaps 2D** (images de densité) qui sont ensuite traitées par des convolutions 3D pour capturer la dynamique temporelle.

> **Signes reconnus :** `OUI`, `NON`, `APPELER`, `VITE`, `MANGER`, `COMPRENDRE`, `BOIRE`, `MALADE.VENTRE`, `CHAUD`, `DORMIR`, `RESPIRER`, `TOMBER`, `FROID`, `FAIBLE`, `ENCEINTE`, `HOPITAL`, `SOUFFRIR`, `PAS.SOUFFLER`, `FORT`, `MEDECIN`.

---

## 🛠️ Installation & Utilisation

```bash
# Installation des dépendances
pip install -r requirements.txt

# Inférence Temps Réel (via Webcam)
python src/utils/inference.py
```

*   **Contrôles** : `ESPACE` pour réinitialiser le buffer de détection, `Q` pour quitter.
*   **Fonctionnement** : Le système utilise MediaPipe Holistic pour détecter les mains et le corps. La prédiction est stabilisée par un vote majoritaire sur les frames accumulées.

---

## 🤖 Autres Expérimentations (Branche Spoter)

Nous avons également testé une approche basée sur les Transformers : **SPOTER** (Sign Pose-Based Transformer). Cette version utilise le mécanisme d'attention pour traiter les séquences de signes.

Cette approche est isolée dans la branche et le dossier `spoter/`. Pour plus de détails sur son fonctionnement, son entraînement et sa normalisation spécifique (Bohacek), veuillez consulter le **[README dédié à SPOTER](./spoter/README.md)**.

---

## 📝 Maintenance & Données

*   **Ré-extraction des poses** : `python src/utils/extract_poses.py` pour aligner les données sur la version actuelle de MediaPipe.
*   **Dataset source** : Les données proviennent de [LSFB ISOL v2](https://lsfb.info.unamur.be).
