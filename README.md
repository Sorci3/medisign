# MediSign Assist

**Real-time French-Belgian Sign Language (LSFB) recognition for medical emergencies.**

When a deaf or non-verbal patient meets a paramedic who doesn't sign, the first casualty is the diagnosis. MediSign Assist recognises 20 medical-emergency signs from a webcam or an Android phone, so a care team can understand and act immediately — while waiting for a human interpreter.


> **Status:** research prototype. It works, and it is **not** fit for clinical use — see [Honest limitations](#honest-limitations). We would rather say that up front than have you find out the hard way.

---

## Results

We trained and compared three architectures on skeletal keypoints. SPOTER wins, and it is the one shipped on this branch.

| Model | Test accuracy | Loss | Training time | Where |
|---|---|---|---|---|
| **SPOTER** (Transformer, fine-tuned) | **83.4 %** | ~1.4 | 1h20 pre-train + 20 min fine-tune | this branch |
| TCN (self-supervised pre-training) | ~60 % | ~1.6 | 40 min | [`TCN---I3D`](../../tree/TCN---I3D) |
| I3D (inflated 3D ConvNet) | ~30 % | ~3.5 | ~2 h | [`TCN---I3D`](../../tree/TCN---I3D) |

SPOTER's 83.4 % comes from [`results/run_20260416_193550/report.txt`](results/run_20260416_193550/report.txt) and is recorded in `models/spoter/meta.json`. Every training run writes its own learning curves, confusion matrix and classification report to `results/`.

**Read that number carefully.** Weighted F1 is 0.85, but **macro F1 is only 0.62**. The dataset is severely imbalanced: the test set holds 346 `OUI` samples and exactly 1 `RESPIRER`. Frequent signs are genuinely well recognised (`OUI`, `NON` and `COMPRENDRE` all exceed 0.90 F1); rare signs are barely evaluable. The imbalance — not the architecture — is the ceiling on this project.

### Why SPOTER wins

I3D was our first attempt, chosen because the LSFB paper used it. We fed it skeleton heatmaps instead of video to keep training tractable on our own machines, and it massively overfit (93 % train vs 30 % val): I3D draws its power from video pre-training we had no weights for. The TCN fixed the overfitting via SignBERT-style self-supervised pre-training, but plateaued around 60 %. SPOTER was designed for sign language from the start — attention over the whole sequence, plus a geometric normalisation that makes coordinates independent of the signer's position and body shape.

---

## The 20 signs

Chosen to cover the situations a first responder actually faces:

| Category | Signs |
|---|---|
| Communication | `OUI` · `NON` · `COMPRENDRE` |
| Vital needs | `MANGER` · `BOIRE` · `DORMIR` |
| Emergency | `APPELER` · `VITE` |
| Symptoms | `CHAUD` · `FROID` · `SOUFFRIR` · `FORT` · `FAIBLE` · `MALADE.VENTRE` · `PAS.SOUFFLER` · `RESPIRER` · `ENCEINTE` |
| Mobility | `TOMBER` |
| Medical | `HOPITAL` · `MEDECIN` |

---

## How it works

```
Webcam ──▶ MediaPipe ──▶ Bohácek ──▶ SPOTER ──▶ Top-5
  video    PoseLandmarker (33)   normalisation  Transformer   prediction
           HandLandmarker (21×2)                encoder-decoder
                    │                                 │
              225 coordinates                   ~750k parameters
              per frame
```

We work on **coordinates, not raw pixels**. That decision is what made the project feasible: it cut compute enough to train everything on our own hardware.

**Normalisation (Bohácek).** Body coordinates are recentred on the mid-shoulder point and scaled by inter-shoulder distance; each hand gets a bounding box recomputed every frame. The model then only sees the *shape of the gesture*, not where the signer stands or how tall they are.

**Two-phase training.** Training SPOTER directly on 20 signs hit the dataset-size wall immediately. So: pre-train on every LSFB class with more than 50 instances, then fine-tune on our 20 medical signs. Pre-training alone reaches ~40 %; fine-tuning takes it to 83 %.

**Architecture.** Each frame is projected 225 → 64 dims, gets a learned positional bias plus a positional encoding, then passes through 6 multi-head self-attention layers (4 heads). The decoder uses a single learned `class_query` token — a [CLS] analogue — which queries the encoded sequence through 6 cross-attention layers to produce one global gesture vector, projected to 20 classes. Decoder self-attention is removed: it does nothing on a single token.

---

## Quick start

```bash
pip install -r requirements.txt
python inference.py
```

Runs live webcam recognition with the trained model in `models/spoter/`. It collects 60 frames (~2 s), predicts, then resets automatically. Top-5 candidates are displayed with confidence bars. Press `c` to reset the buffer manually, `q` to quit.

<details>
<summary><b>Full pipeline: data preparation and training</b></summary>

### 1. Data

```bash
python prepare_data.py --step download            # 20 medical signs
python prepare_data.py --step pretrain-download   # full LSFB for pre-training
python prepare_data.py --step pretrain-landmarks
python prepare_data.py --step index
python prepare_data.py --step landmarks
```

Data lands in the folder set by `dataset_path` in `config.json`.

### 2. Pre-training (generalist)

```bash
python pretrain.py
```

Learns the structure of signs across the whole LSFB dataset.

### 3. Fine-tuning (specialist)

```bash
python train.py --pretrained models/spoter/pretrained.pt
```

Freezes the backbone for 10 epochs to stabilise the classification head, then unfreezes the whole network.

</details>

All hyperparameters live in [`config.json`](config.json) — `hidden_dim` 64, 4 heads, 6 encoder / 6 decoder layers, 225 input features.

---

## Android app

The SPOTER model is embedded in a working Android app, with a live landmark overlay and front/rear camera switching. The hard part was porting the normalisation logic from Python to Kotlin, since several libraries we relied on simply don't exist there.

Accuracy on-device is **below** our desktop results: even on the phone GPU, MediaPipe inference plus sign inference introduces latency that degrades keypoint extraction quality. Optimising the normalisation path is the obvious next step.

> The app source is not yet published in this repository.

---

## Honest limitations

We would rather document these than let a demo video imply otherwise.

- **Latency.** The buffer needs a couple of seconds to capture a sign and predict. In a real emergency, that is too slow.
- **Wrong predictions carry real risk.** A misread sign in a medical context can endanger a patient. Accuracy is *correct*, not *sufficient*.
- **Dataset imbalance.** Some signs have ~10 natural instances. No amount of augmentation manufactures information that was never recorded.
- **Fixed-window buffer.** We assume a sign fits a fixed window rather than detecting where it starts and ends.

**Where we'd go next:** a dataset enriched by actual sign language professionals, and a start/end-of-sequence detection model to replace the fixed buffer.

---

## Contributing

The gap between "83 % on a benchmark" and "trustworthy in an ambulance" is wide, and we would genuinely welcome help closing it. Good places to start:

- **Rare-sign performance** — the macro F1 of 0.62 is the single biggest weakness. Better augmentation, class-balanced losses, or few-shot approaches.
- **Sequence segmentation** — detect sign boundaries instead of using a fixed 60-frame buffer.
- **On-device optimisation** — the Kotlin normalisation path is where the mobile accuracy drop lives.
- **Data** — if you sign LSFB, or work with people who do, that is the contribution we need most.

Open an issue to discuss an idea, or send a pull request. Questions and reproduction attempts are welcome too — if something in here doesn't reproduce, we want to know.

One constraint to know before you start: the LSFB corpus is non-commercial, so anything built on our weights or the dataset stays non-commercial too. See [License](#license).

---

## Repository layout

| Branch | Contents |
|---|---|
| [`main`](../../tree/main) | SPOTER — best results, live inference, shipped model |
| [`TCN---I3D`](../../tree/TCN---I3D) | TCN and I3D experiments, preprocessing and training notebooks |

---

## Team

Built by five AI students.

| GitHub |
|---|
| [@Sorci3](https://github.com/Sorci3) |
| [@matili0](https://github.com/matili0) AKA Herbreteau Mathis|
| [@Gobx1](https://github.com/Gobx1) AKA Bonneau Axel|
| [@MamatorHack](https://github.com/MamatorHack) |
| Louis Maillet |

---

## License

This repository is split across two licenses. Please read both before reusing anything.

| What | License |
|---|---|
| **Our code** (`*.py`, `config.json`) | [MIT](LICENSE) |
| **Trained weights** (`models/spoter/*.pt`) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — inherited from the dataset |

The weights are trained on the LSFB ISOL corpus, which is distributed under **CC BY-NC-SA 4.0**. We treat the resulting models as derivatives of that corpus, so the dataset's terms follow them. In practice, if you use our weights or the dataset:

- **No commercial use.**
- **Share alike** — redistribute derivatives under CC BY-NC-SA 4.0.
- **Attribute** — cite the [LSFB website](https://lsfb.info.unamur.be/#dataset) and the two citations in [Dataset citation](#dataset-citation) below.
- **Don't add restrictions** beyond what the license permits.

Our own code is MIT, so you are free to reuse the architecture, the normalisation and the training pipeline commercially — but train it on data you are entitled to use, not on ours.

---

## Dataset citation

The LSFB dataset providers require these citations for any use of the corpus. If you use this repository's weights, they apply to you too.

> Meurant, Laurence (2015). *Corpus LSFB. Un corpus informatisé en libre accès de vidéos et d'annotations de la langue des signes de Belgique francophone (LSFB).* Laboratoire de Langue des signes de Belgique francophone (LSFB-Lab). FRS-F.N.R.S et Université de Namur.

And the accompanying article: [ieeexplore.ieee.org/abstract/document/9534336](https://ieeexplore.ieee.org/abstract/document/9534336)

Dataset home: [lsfb.info.unamur.be](https://lsfb.info.unamur.be/#dataset)

---

## References

**Architecture**

- Bohácek, M., & Hrúz, M. (2022). [Sign Pose-based Transformer for Word-level Sign Language Recognition](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html). *IEEE/CVF WACV Workshops*, 182–191. — the architecture this branch implements.
- Basso Madjoukeng, A., et al. [SSL-SLR: Self-Supervised Representation Learning for Sign Language Recognition](https://arxiv.org/abs/2509.05188). arXiv:2509.05188.

**Dataset** — see [Dataset citation](#dataset-citation) for the citations the corpus licence requires.

- Meurant, L. (2015). *Corpus LSFB*. LSFB-Lab, FRS-F.N.R.S & Université de Namur.
- [LSFB-CONT and LSFB-ISOL: Two New Datasets for Vision-Based Sign Language Recognition](https://ieeexplore.ieee.org/abstract/document/9534336). IEEE.
- Fink, J. (2020). [Free datasets for Sign Language Recognition — LSFB](https://lsfb.info.unamur.be/#dataset). University of Namur.

**Data augmentation**

- Rios, G. G., et al. (2025). HandCraft: Dynamic Sign Generation for Synthetic Data Augmentation. [doi:10.48550/arXiv.2508.14345](https://doi.org/10.48550/arXiv.2508.14345)
- Madjoukeng, A., et al. (2025). Benchmarking Data Augmentation. [doi:10.14428/esann/2025.ES2025-142](https://doi.org/10.14428/esann/2025.ES2025-142)
- Madjoukeng, A., et al. (2025). Local-global Data Augmentation for Contrastive Learning. [doi:10.1007/978-3-031-91398-3_5](https://doi.org/10.1007/978-3-031-91398-3_5)
