# TCN Preprocessing — Design Spec
Date: 2026-04-07

## Context

The TCN approach suffers from severe class imbalance (OUI: 1291 instances, MEDECIN: 9) and insufficient training data. This spec covers:
1. A new notebook `src/notebooks/01_tcn_preprocessing.ipynb` that applies data augmentation from research papers and saves an augmented dataset to a new folder.
2. Modifications to `src/notebooks/02_tcn_training.ipynb` to remove its built-in augmentation and load from the pre-processed dataset instead.

## Data Format

**Input**: Raw pose files in `src/dataset/poses/{pose, left_hand, right_hand}/`, each `(T, K, 3)` float32.  
**Output**: Flattened coordinate vectors `(225, 32)` = (33 body + 21 left_hand + 21 right_hand) × 3 coords, 32 frames. Same format as `CoordDataset` in the training notebook.

Only the **train split** is augmented. The test set is never touched.

## Output Structure

```
src/dataset/augmented_tcn/
├── npy/
│   ├── <INSTANCE_ID>_orig.npy        # (225, 32) float32
│   ├── <INSTANCE_ID>_aug00.npy
│   ├── <INSTANCE_ID>_aug01.npy
│   ...
├── metadata.csv                      # id, sign, label, source_id, aug_type
└── sign_to_idx.json
```

## Augmentation Strategy

**Target**: all classes reach 1291 instances (= count of OUI, the majority class).  
**Cap**: max 50 augmentation variants per original instance. If a class needs more than `50 × n_original` variants, repeat variants with new random parameters.

### Augmentation Functions (operate on raw `(T, K, 3)` poses before resampling to 32 frames)

| Function | Description | Source |
|----------|-------------|--------|
| `aug_horizontal_flip` | Flip x-coord (1−x), swap left/right hands | Benchmarking_Data paper |
| `aug_temporal_flip` | Reverse frame order | General SLR practice |
| `aug_global_translation` | Shift all keypoints (Δx, Δy) ∈ [−0.1, 0.1] | Local-global paper |
| `aug_global_scaling` | Scale ±10% around skeleton center | handcraft paper |
| `aug_global_rotation` | 2D rotation ±5° around center | handcraft paper |
| `aug_speed_perturbation` | Stretch/compress temporal axis ±20% before resampling | handcraft paper (temporal) |

### Fixed Combination Catalogue (12 deterministic variants)

| Index | Combination |
|-------|-------------|
| 0 | horizontal flip |
| 1 | temporal flip |
| 2 | translation (random) |
| 3 | scaling (random) |
| 4 | rotation (random) |
| 5 | speed perturbation (random) |
| 6 | flip + translation |
| 7 | flip + scaling |
| 8 | flip + rotation |
| 9 | temporal flip + speed |
| 10 | translation + scaling + rotation |
| 11 | flip + temporal flip + jitter |

If 12 × n_original < target, additional variants are generated with fresh random parameters (new translation/scale/rotation draws each time).

## Notebook Structure — `01_tcn_preprocessing.ipynb`

| Cell | Content |
|------|---------|
| 1 | Imports, paths, constants: `TARGET = 1291`, `MAX_VARIANTS_PER_INSTANCE = 50`, `TARGET_T = 32` |
| 2 | Load instances, class distribution plot |
| 3 | Remove T=0 instances, load train/test splits, print stats |
| 4 | `resample_sequence` + `load_raw_poses` utilities |
| 5 | 6 augmentation functions |
| 6 | `poses_to_coords`: converts `(T, K, 3)` → `(225, 32)` flattened vector |
| 7 | Combination catalogue definition |
| 8 | Main balancing loop: for each class, generate variants until 1291 reached |
| 9 | Save `.npy` files, `metadata.csv`, `sign_to_idx.json` to `augmented_tcn/` |
| 10 | Verification: final class distribution plot + shape sanity checks |

## Modifications to `02_tcn_training.ipynb`

**Remove:**
- Cell 4: all 4 augmentation functions (`augment_horizontal_flip`, `augment_temporal_flip`, `augment_jitter`, `augment_time_warp`)
- `augment` parameter and all augmentation logic from `CoordDataset.__getitem__`
- `WeightedRandomSampler` (dataset is already balanced)

**Modify:**
- `CoordDataset` loads from two sources:
  - **Train**: reads `augmented_tcn/npy/<id>.npy` using `metadata.csv`
  - **Test**: continues loading from `poses/` via `load_coords` as before
- `train_loader` uses `shuffle=True` instead of the sampler
