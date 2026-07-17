# MediSign Assist — TCN & I3D experiments

> **This is not the main branch.** It holds the two convolutional architectures we tried *before* SPOTER, both of which SPOTER beat. For the project overview, the shipped model and the Android app, go to **[`main`](../../tree/main)**.

We keep this branch because negative results are results. It documents two approaches that did not work, and why — which is most of what we learned.

| Model | Accuracy | Loss | Training time |
|---|---|---|---|
| I3D (inflated 3D ConvNet) | ~30 % | ~3.5 | ~2 h |
| TCN (self-supervised pre-training) | ~60 % | ~1.6 | 40 min |
| **SPOTER** — on [`main`](../../tree/main) | **83.4 %** | ~1.4 | 1h20 + 20 min |

Both models here operate on skeletal keypoints extracted with MediaPipe from the [LSFB ISOL](https://lsfb.info.unamur.be/#dataset) corpus, across the same 20 medical-emergency signs used on `main`.

---

## I3D — Inflated 3D ConvNet

We started here because I3D is the architecture referenced in the LSFB dataset paper. It inflates a 2D network pre-trained on images (Inception-V1) into 3D by turning every H×W convolution into a T×H×W one, adding a temporal dimension.

We did not want to train on video — the cost would have been an order of magnitude beyond our hardware — so we rendered the skeleton coordinates into **2D heatmaps** and fed those instead. Input tensors are `B × 3 × 32 × 64 × 64`: three channels (body, left hand, right hand), 32 temporal frames, 64×64 px.

**It failed, and we know why.** Training accuracy hit 93 % while validation sat at 30 %, with an unstable, high validation loss and an F1 of roughly 0.16. Two causes:

1. **I3D was never designed for heatmaps.** It expects video. Feeding it density images of coordinates strips away exactly the texture its convolutions are built to exploit.
2. **No pre-training.** I3D draws nearly all its power from large-scale video pre-training, and no pre-trained weights exist for coordinate heatmaps. We were training from scratch on a small, imbalanced dataset.

---

## TCN — Temporal Convolutional Network

The TCN drops heatmaps entirely and works directly on keypoint coordinates: each sequence is a `(225, 32)` tensor — 225 coordinates (33 body + 21 left hand + 21 right hand points) resampled to 32 frames.

It trains in two phases, borrowing from transfer learning in NLP and from self-supervision in sign language recognition (SignBERT).

**Phase 1 — self-supervised pre-training.** 30 % of each sequence's frames are replaced by a learned mask token, and the TCN backbone reconstructs the original features at the masked positions through a light decoder (MSE loss). No labels needed, so this exploits the *entire* LSFB-ISOL corpus (~120k sequences) rather than just our 20 signs.

**Phase 2 — supervised fine-tuning.** The pre-trained encoder is frozen during warm-up, then an MLP head (LayerNorm → Dropout 0.5 → Linear 256→128 + GELU → Dropout 0.25 → Linear 128→20) is trained on the medical signs, with label smoothing and mixup.

### The interesting part

We trained both a baseline TCN and a pre-trained one, and the result is not the clean story we expected:

| Variant | Train acc | Val acc | F1 | Behaviour |
|---|---|---|---|---|
| Baseline | ~100 % | ~75 % | ~0.50 | Overfits badly; validation loss unstable and far above train loss |
| Pre-trained | ~40 % | ~60 % | ~0.40 | Does not overfit; validation loss stable and tracking train loss |

Self-supervised pre-training did exactly what it was supposed to do — **it killed the overfitting** — but it did *not* buy us better validation accuracy. The baseline still scores higher on both val accuracy and F1, it just gets there by memorising. We report the pre-trained TCN as our ~60 % result because its numbers are the ones we trust, not the ones we like.

Either way, both sit well below SPOTER. An architecture built for sign language beat the one we adapted to it.

---

## Layout

```
src/
├── notebooks/
│   ├── 01_i3d_preprocessing.ipynb   # coordinates → heatmaps
│   ├── 01_tcn_preprocessing.ipynb   # coordinates → (225, 32) tensors
│   ├── 02_i3d_training.ipynb
│   ├── 02_tcn_training.ipynb
│   └── 03_tcn_pretrain.ipynb        # self-supervised masking phase
├── utils/
│   ├── inference.py                 # live webcam — pre-trained TCN
│   ├── inference_baseline.py        # live webcam — baseline TCN
│   ├── extract_poses.py             # re-extract poses with current MediaPipe
│   ├── dataset_extraction.py
│   ├── download_pretrain_data.py
│   └── config.json
├── models/                          # trained weights (.pth) + MediaPipe tasks
└── signs.py                         # the 20 target signs, single source of truth
```

## Usage

```bash
pip install -r requirements.txt
python src/utils/inference.py
```

Live webcam recognition with the pre-trained TCN. MediaPipe Holistic detects body and hands; predictions are stabilised by majority vote over the accumulated frames. Press `SPACE` to reset the detection buffer, `Q` to quit.

To reproduce training, run the notebooks in order: preprocessing → (pre-training) → training. To realign the extracted poses with your installed MediaPipe version, run `python src/utils/extract_poses.py`.

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

Same terms as [`main`](../../tree/main): our code is [MIT](LICENSE), and the trained weights in `src/models/*.pth` are derived from the LSFB ISOL corpus and therefore fall under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — **non-commercial use only**, share-alike, attribution required.

Using the dataset or these weights requires citing:

> Meurant, Laurence (2015). *Corpus LSFB. Un corpus informatisé en libre accès de vidéos et d'annotations de la langue des signes de Belgique francophone (LSFB).* Laboratoire de Langue des signes de Belgique francophone (LSFB-Lab). FRS-F.N.R.S et Université de Namur.

and [LSFB-CONT and LSFB-ISOL: Two New Datasets for Vision-Based Sign Language Recognition](https://ieeexplore.ieee.org/abstract/document/9534336), plus the [LSFB website](https://lsfb.info.unamur.be/#dataset).

## References

- Basso Madjoukeng, A., et al. [SSL-SLR: Self-Supervised Representation Learning for Sign Language Recognition](https://arxiv.org/abs/2509.05188). arXiv:2509.05188. — the self-supervision approach behind our TCN pre-training.
- Full bibliography on [`main`](../../tree/main#references).
