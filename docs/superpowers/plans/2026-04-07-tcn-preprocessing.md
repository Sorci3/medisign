# TCN Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `src/notebooks/01_tcn_preprocessing.ipynb` with paper-based augmentation that balances all 20 classes to 1291 instances, saves to `src/dataset/augmented_tcn/`, and remove the built-in augmentation from `02_tcn_training.ipynb`.

**Architecture:** The preprocessing notebook loads raw pose files `(T, K, 3)`, applies 6 augmentation transforms in 12 fixed combinations, then saves flattened `(225, 32)` coordinate vectors. The training notebook is simplified to load pre-computed arrays from `augmented_tcn/npy/` for train and original poses for test.

**Tech Stack:** Python 3.12, NumPy, Pandas, tqdm, Jupyter / NotebookEdit tool

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/notebooks/01_tcn_preprocessing.ipynb` | Full preprocessing pipeline |
| Modify | `src/notebooks/02_tcn_training.ipynb` | Remove augmentation, load from augmented_tcn/ |
| Create (output) | `src/dataset/augmented_tcn/npy/*.npy` | Pre-computed (225, 32) arrays |
| Create (output) | `src/dataset/augmented_tcn/metadata.csv` | id, sign, label, source_id, aug_type |
| Create (output) | `src/dataset/augmented_tcn/sign_to_idx.json` | Label mapping |

---

## Task 1: Notebook skeleton — imports, paths, data loading (cells 1–3)

**Files:**
- Create: `src/notebooks/01_tcn_preprocessing.ipynb`

> Use the `NotebookEdit` tool to add cells. The file currently exists but is empty/malformed — overwrite it with a fresh notebook.

- [ ] **Step 1: Add cell 1 — imports and constants**

```python
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm

DATASET_ROOT = Path("../dataset")
POSES_ROOT   = DATASET_ROOT / "poses"
METADATA_DIR = DATASET_ROOT / "metadata"
AUG_ROOT     = DATASET_ROOT / "augmented_tcn"

SIGNS_TARGET = [
    "SOUFFRIR", "AIDER",    "FORT",     "MALADE",   "COEUR",
    "TETE",     "MORT",     "PLEURER",      "NON",   "FROID",
    "MANGER",    "OUI",   "TOMBER", "ACCIDENT", "MARCHER",
    "ENCEINTE", "DORMIR",  "BOIRE",     "CHAUD",  "MEDECIN"
]
SIGN_TO_IDX = {s: i for i, s in enumerate(SIGNS_TARGET)}
NUM_CLASSES  = len(SIGNS_TARGET)
TARGET_T     = 32

TARGET_COUNT          = 1291   # match OUI (majority class)
MAX_VARIANTS_PER_ORIG = 50     # cap per original instance

print(f"Classes: {NUM_CLASSES} | Target per class: {TARGET_COUNT} | Max variants/orig: {MAX_VARIANTS_PER_ORIG}")
```

- [ ] **Step 2: Add cell 2 — load instances, plot class distribution**

```python
instances = pd.read_csv(DATASET_ROOT / "instances.csv")
face_files = list((POSES_ROOT / "face").glob("*.npy"))
available_ids = {f.stem for f in face_files}

df = instances[instances["id"].isin(available_ids)].copy()
df["label"] = df["sign"].map(SIGN_TO_IDX)
df = df[df["label"].notna()].copy()
df["label"] = df["label"].astype(int)

print(f"Total instances with poses: {len(df)}")
print("\nClass distribution:")
print(df["sign"].value_counts().to_string())

fig, ax = plt.subplots(figsize=(12, 4))
df["sign"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Instance count per sign (raw)")
ax.set_xlabel("Sign")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Add cell 3 — remove T=0 instances, load splits**

```python
# Remove T=0 instances
lengths = {}
for _, row in df.iterrows():
    arr = np.load(POSES_ROOT / "face" / f"{row['id']}.npy")
    lengths[row["id"]] = arr.shape[0]
df["T"] = df["id"].map(lengths)
df_clean = df[df["T"] > 0].reset_index(drop=True)
print(f"Removed {len(df) - len(df_clean)} instances with T=0")
print(f"Remaining: {len(df_clean)}")

# Load official splits
with open(METADATA_DIR / "splits" / "train.json") as f:
    train_ids = set(json.load(f))
with open(METADATA_DIR / "splits" / "test.json") as f:
    test_ids = set(json.load(f))

df_train = df_clean[df_clean["id"].isin(train_ids)].reset_index(drop=True)
df_test  = df_clean[df_clean["id"].isin(test_ids)].reset_index(drop=True)

print(f"\nTrain: {len(df_train)} | Test: {len(df_test)}")
print("\nTrain class distribution:")
print(df_train["sign"].value_counts().to_string())
```

- [ ] **Step 4: Run cells 1–3 and verify no errors, distribution matches known counts (OUI: 1291, MEDECIN: 9)**

- [ ] **Step 5: Commit**

```bash
git add src/notebooks/01_tcn_preprocessing.ipynb
git commit -m "feat: add TCN preprocessing notebook skeleton (cells 1-3)"
```

---

## Task 2: Utility functions — resample_sequence, load_raw_poses (cell 4)

**Files:**
- Modify: `src/notebooks/01_tcn_preprocessing.ipynb`

- [ ] **Step 1: Add cell 4 — resample_sequence and load_raw_poses**

```python
def resample_sequence(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Resample (T, K, 3) array to target_T frames via linear interpolation."""
    T = arr.shape[0]
    if T == target_T:
        return arr
    idx = np.linspace(0, T - 1, target_T)
    lo  = np.floor(idx).astype(int).clip(0, T - 1)
    hi  = np.ceil(idx).astype(int).clip(0, T - 1)
    a   = (idx - lo)[:, None, None]
    return ((1 - a) * arr[lo] + a * arr[hi]).astype(np.float32)


def load_raw_poses(instance_id: str):
    """Return (body, lhand, rhand) as float32 (T, K, 3) arrays — no resampling."""
    body  = np.load(POSES_ROOT / "pose"       / f"{instance_id}.npy").astype(np.float32)
    lhand = np.load(POSES_ROOT / "left_hand"  / f"{instance_id}.npy").astype(np.float32)
    rhand = np.load(POSES_ROOT / "right_hand" / f"{instance_id}.npy").astype(np.float32)
    return body, lhand, rhand


# Sanity check
_sid = df_train["id"].iloc[0]
_b, _l, _r = load_raw_poses(_sid)
assert _b.ndim == 3 and _b.shape[2] == 3, f"Expected (T,K,3), got {_b.shape}"
assert _l.shape[1] == 21,  f"Expected 21 left-hand kpts, got {_l.shape[1]}"
assert _r.shape[1] == 21,  f"Expected 21 right-hand kpts, got {_r.shape[1]}"
_r32 = resample_sequence(_b, 32)
assert _r32.shape[0] == 32, f"Expected 32 frames, got {_r32.shape[0]}"
print(f"load_raw_poses OK: body={_b.shape}, lhand={_l.shape}, rhand={_r.shape}")
print(f"resample_sequence OK: {_b.shape} -> {_r32.shape}")
```

- [ ] **Step 2: Run cell 4 and verify assertions pass**

- [ ] **Step 3: Commit**

```bash
git add src/notebooks/01_tcn_preprocessing.ipynb
git commit -m "feat: add resample_sequence and load_raw_poses utilities"
```

---

## Task 3: Augmentation functions (cell 5)

**Files:**
- Modify: `src/notebooks/01_tcn_preprocessing.ipynb`

All functions operate on raw `(T, K, 3)` float32 arrays. They return `(body, lhand, rhand)` tuples.

- [ ] **Step 1: Add cell 5 — 6 augmentation functions**

```python
def aug_horizontal_flip(body, lhand, rhand):
    """Flip x-coord (1-x) and swap left/right hands. Source: Benchmarking_Data paper."""
    def flip_x(arr):
        arr = arr.copy()
        arr[:, :, 0] = 1.0 - arr[:, :, 0]
        return arr
    return flip_x(body), flip_x(rhand), flip_x(lhand)  # note: L/R swapped


def aug_temporal_flip(body, lhand, rhand):
    """Reverse frame order."""
    return body[::-1].copy(), lhand[::-1].copy(), rhand[::-1].copy()


def aug_global_translation(body, lhand, rhand, dx: float, dy: float):
    """Shift all keypoints by (dx, dy). Values in [-0.1, 0.1]. Source: Local-global paper."""
    def shift(arr):
        arr = arr.copy()
        arr[:, :, 0] = np.clip(arr[:, :, 0] + dx, 0.0, 1.0)
        arr[:, :, 1] = np.clip(arr[:, :, 1] + dy, 0.0, 1.0)
        return arr
    return shift(body), shift(lhand), shift(rhand)


def aug_global_scaling(body, lhand, rhand, scale: float):
    """Scale all keypoints around joint centroid by factor. Source: handcraft paper."""
    all_x = np.concatenate([body[:, :, 0].ravel(), lhand[:, :, 0].ravel(), rhand[:, :, 0].ravel()])
    all_y = np.concatenate([body[:, :, 1].ravel(), lhand[:, :, 1].ravel(), rhand[:, :, 1].ravel()])
    cx, cy = all_x.mean(), all_y.mean()
    def scale_arr(arr):
        arr = arr.copy()
        arr[:, :, 0] = np.clip((arr[:, :, 0] - cx) * scale + cx, 0.0, 1.0)
        arr[:, :, 1] = np.clip((arr[:, :, 1] - cy) * scale + cy, 0.0, 1.0)
        return arr
    return scale_arr(body), scale_arr(lhand), scale_arr(rhand)


def aug_global_rotation(body, lhand, rhand, angle_deg: float):
    """2D rotation around skeleton centroid. Source: handcraft paper (±5°)."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    all_x = np.concatenate([body[:, :, 0].ravel(), lhand[:, :, 0].ravel(), rhand[:, :, 0].ravel()])
    all_y = np.concatenate([body[:, :, 1].ravel(), lhand[:, :, 1].ravel(), rhand[:, :, 1].ravel()])
    cx, cy = all_x.mean(), all_y.mean()
    def rotate_arr(arr):
        arr = arr.copy()
        x = arr[:, :, 0] - cx
        y = arr[:, :, 1] - cy
        arr[:, :, 0] = np.clip(cos_t * x - sin_t * y + cx, 0.0, 1.0)
        arr[:, :, 1] = np.clip(sin_t * x + cos_t * y + cy, 0.0, 1.0)
        return arr
    return rotate_arr(body), rotate_arr(lhand), rotate_arr(rhand)


def aug_speed_perturbation(body, lhand, rhand, factor: float):
    """Stretch/compress temporal axis by factor before resampling. Source: handcraft paper."""
    T = body.shape[0]
    new_T = max(1, int(round(T * factor)))
    idx = np.linspace(0, T - 1, new_T)
    lo  = np.floor(idx).astype(int).clip(0, T - 1)
    hi  = np.ceil(idx).astype(int).clip(0, T - 1)
    a   = (idx - lo)[:, None, None]
    interp = lambda arr: ((1 - a) * arr[lo] + a * arr[hi]).astype(np.float32)
    return interp(body), interp(lhand), interp(rhand)


# Sanity checks
_b, _l, _r = load_raw_poses(df_train["id"].iloc[0])
_bf, _lf, _rf = aug_horizontal_flip(_b, _l, _r)
assert _bf.shape == _b.shape and np.all(_bf[:, :, 0] == 1.0 - _b[:, :, 0]), "hflip failed"
_bt, _lt, _rt = aug_temporal_flip(_b, _l, _r)
assert np.allclose(_bt[0], _b[-1]), "temporal flip failed"
_bs, _ls, _rs = aug_speed_perturbation(_b, _l, _r, 1.2)
assert _bs.shape[0] == max(1, int(round(_b.shape[0] * 1.2))), "speed perturbation failed"
print("All augmentation functions OK")
```

- [ ] **Step 2: Run cell 5 and verify all assertions pass**

- [ ] **Step 3: Commit**

```bash
git add src/notebooks/01_tcn_preprocessing.ipynb
git commit -m "feat: add 6 augmentation functions (flip, temporal, translation, scaling, rotation, speed)"
```

---

## Task 4: poses_to_coords and combination catalogue (cells 6–7)

**Files:**
- Modify: `src/notebooks/01_tcn_preprocessing.ipynb`

- [ ] **Step 1: Add cell 6 — poses_to_coords**

```python
def poses_to_coords(body, lhand, rhand, target_T: int = TARGET_T) -> np.ndarray:
    """
    Resample and flatten raw poses into (225, target_T) float32 array.
    Layout: 99 body coords (33 kpts × 3) + 63 lhand + 63 rhand = 225 rows.
    """
    body  = resample_sequence(body,  target_T)  # (target_T, 33, 3)
    lhand = resample_sequence(lhand, target_T)  # (target_T, 21, 3)
    rhand = resample_sequence(rhand, target_T)  # (target_T, 21, 3)

    body_flat  = body.reshape(target_T, -1)   # (target_T, 99)
    lhand_flat = lhand.reshape(target_T, -1)  # (target_T, 63)
    rhand_flat = rhand.reshape(target_T, -1)  # (target_T, 63)

    coords = np.concatenate([body_flat, lhand_flat, rhand_flat], axis=1)  # (target_T, 225)
    return coords.T.astype(np.float32)  # (225, target_T) channel-first for Conv1d


# Sanity check
_b, _l, _r = load_raw_poses(df_train["id"].iloc[0])
_coords = poses_to_coords(_b, _l, _r)
assert _coords.shape == (225, TARGET_T), f"Expected (225, {TARGET_T}), got {_coords.shape}"
assert _coords.dtype == np.float32, f"Expected float32, got {_coords.dtype}"
print(f"poses_to_coords OK: shape={_coords.shape}, range=[{_coords.min():.3f}, {_coords.max():.3f}]")
```

- [ ] **Step 2: Add cell 7 — combination catalogue and apply_combo**

```python
COMBINATIONS = [
    "hflip",
    "tflip",
    "translate",
    "scale",
    "rotate",
    "speed",
    "hflip+translate",
    "hflip+scale",
    "hflip+rotate",
    "tflip+speed",
    "translate+scale+rotate",
    "hflip+tflip+translate",
]


def apply_combo(body, lhand, rhand, combo: str):
    """Apply a named combination of augmentations. Random params drawn fresh each call."""
    b, l, r = body, lhand, rhand
    if "hflip" in combo:
        b, l, r = aug_horizontal_flip(b, l, r)
    if "tflip" in combo:
        b, l, r = aug_temporal_flip(b, l, r)
    if "translate" in combo:
        dx = np.random.uniform(-0.1, 0.1)
        dy = np.random.uniform(-0.1, 0.1)
        b, l, r = aug_global_translation(b, l, r, dx, dy)
    if "scale" in combo:
        s = np.random.uniform(0.9, 1.1)
        b, l, r = aug_global_scaling(b, l, r, s)
    if "rotate" in combo:
        angle = np.random.uniform(-5.0, 5.0)
        b, l, r = aug_global_rotation(b, l, r, angle)
    if "speed" in combo:
        factor = np.random.uniform(0.8, 1.2)
        b, l, r = aug_speed_perturbation(b, l, r, factor)
    return b, l, r


# Quick sanity: apply all 12 combos on one instance
_b, _l, _r = load_raw_poses(df_train["id"].iloc[0])
for _combo in COMBINATIONS:
    _ab, _al, _ar = apply_combo(_b, _l, _r, _combo)
    _coords = poses_to_coords(_ab, _al, _ar)
    assert _coords.shape == (225, TARGET_T), f"Combo '{_combo}' produced shape {_coords.shape}"
print(f"apply_combo OK for all {len(COMBINATIONS)} combinations")
```

- [ ] **Step 3: Run cells 6–7 and verify assertions pass**

- [ ] **Step 4: Commit**

```bash
git add src/notebooks/01_tcn_preprocessing.ipynb
git commit -m "feat: add poses_to_coords and 12-combo augmentation catalogue"
```

---

## Task 5: Main balancing loop (cell 8)

**Files:**
- Modify: `src/notebooks/01_tcn_preprocessing.ipynb`

- [ ] **Step 1: Add cell 8 — balancing loop**

```python
# Pre-create output directory
(AUG_ROOT / "npy").mkdir(parents=True, exist_ok=True)

records = []  # will become metadata.csv rows

for sign in tqdm(SIGNS_TARGET, desc="Signs"):
    sign_df    = df_train[df_train["sign"] == sign].reset_index(drop=True)
    label      = SIGN_TO_IDX[sign]
    n_orig     = len(sign_df)
    n_needed   = max(0, TARGET_COUNT - n_orig)

    # --- Save original instances ---
    for _, row in sign_df.iterrows():
        b, l, r = load_raw_poses(row["id"])
        coords  = poses_to_coords(b, l, r)
        fname   = f"{row['id']}_orig"
        np.save(AUG_ROOT / "npy" / f"{fname}.npy", coords)
        records.append({"id": fname, "sign": sign, "label": label,
                        "source_id": row["id"], "aug_type": "original"})

    if n_needed == 0:
        continue

    # --- Generate augmented instances ---
    cap         = MAX_VARIANTS_PER_ORIG * n_orig  # hard cap on total augmented
    n_to_gen    = min(n_needed, cap)
    aug_count   = 0
    combo_cycle = 0  # cycles through COMBINATIONS, repeats if needed

    while aug_count < n_to_gen:
        for _, row in sign_df.iterrows():
            if aug_count >= n_to_gen:
                break
            b, l, r  = load_raw_poses(row["id"])
            combo    = COMBINATIONS[combo_cycle % len(COMBINATIONS)]
            ab, al, ar = apply_combo(b, l, r, combo)
            coords   = poses_to_coords(ab, al, ar)
            fname    = f"{row['id']}_aug{aug_count:03d}"
            np.save(AUG_ROOT / "npy" / f"{fname}.npy", coords)
            records.append({"id": fname, "sign": sign, "label": label,
                            "source_id": row["id"], "aug_type": combo})
            aug_count  += 1
            combo_cycle += 1

print(f"\nTotal records: {len(records)}")
```

- [ ] **Step 2: Run cell 8 — this will take several minutes. Expect a tqdm progress bar per sign. Verify no FileNotFoundError.**

- [ ] **Step 3: Commit**

```bash
git add src/notebooks/01_tcn_preprocessing.ipynb
git commit -m "feat: add balancing loop generating augmented instances to augmented_tcn/"
```

---

## Task 6: Save metadata and verify (cells 9–10)

**Files:**
- Modify: `src/notebooks/01_tcn_preprocessing.ipynb`

- [ ] **Step 1: Add cell 9 — save metadata.csv and sign_to_idx.json**

```python
df_meta = pd.DataFrame(records)
df_meta.to_csv(AUG_ROOT / "metadata.csv", index=False)

with open(AUG_ROOT / "sign_to_idx.json", "w") as f:
    json.dump(SIGN_TO_IDX, f, indent=2)

print("Saved:")
print(f"  metadata.csv  — {len(df_meta)} rows")
print(f"  sign_to_idx.json")
print(f"  npy/ — {len(list((AUG_ROOT / 'npy').glob('*.npy')))} files")
```

- [ ] **Step 2: Add cell 10 — verification and final distribution plot**

```python
# Shape sanity check on 5 random files
import random
npy_files = list((AUG_ROOT / "npy").glob("*.npy"))
random.seed(42)
for f in random.sample(npy_files, min(5, len(npy_files))):
    arr = np.load(f)
    assert arr.shape == (225, TARGET_T), f"{f.name}: expected (225,{TARGET_T}), got {arr.shape}"
    assert arr.dtype == np.float32,       f"{f.name}: expected float32, got {arr.dtype}"
print("Shape sanity: OK")

# Final class distribution
counts_aug = df_meta.groupby("sign").size().reindex(SIGNS_TARGET)
print("\nFinal class counts (augmented train):")
print(counts_aug.to_string())

fig, ax = plt.subplots(figsize=(12, 4))
counts_aug.plot(kind="bar", ax=ax, color="seagreen")
ax.axhline(TARGET_COUNT, color="red", linestyle="--", label=f"target={TARGET_COUNT}")
ax.set_title("Instance count per sign (after augmentation)")
ax.set_xlabel("Sign")
ax.set_ylabel("Count")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Run cells 9–10. Verify all 20 class counts are ≤ TARGET_COUNT (1291), and bars are roughly equal.**

- [ ] **Step 4: Commit**

```bash
git add src/notebooks/01_tcn_preprocessing.ipynb
git commit -m "feat: add save + verification cells for augmented_tcn dataset"
```

---

## Task 7: Modify 02_tcn_training.ipynb — remove augmentation, load from augmented_tcn/

**Files:**
- Modify: `src/notebooks/02_tcn_training.ipynb`

The notebook currently has 13 cells. Cell indices (0-based) to change:
- Cell 3 (index 3): delete entirely — augmentation functions
- Cell 4 (index 4): replace `CoordDataset` — remove augment logic, add augmented path support
- Cell 1 (index 1): add `AUG_ROOT` constant
- Cell 0 (index 0): remove `WeightedRandomSampler` from imports

- [ ] **Step 1: Remove `WeightedRandomSampler` from the import in cell 1**

Replace this line in cell 1:
```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
```
With:
```python
from torch.utils.data import Dataset, DataLoader
```

- [ ] **Step 2: Add `AUG_ROOT` constant at end of cell 1**

Append to cell 1 (after the existing constants):
```python
AUG_ROOT = DATASET_ROOT / "augmented_tcn"
```

- [ ] **Step 3: Delete cell 4 (the augmentation functions cell)**

Use `NotebookEdit` to delete the cell containing `augment_horizontal_flip`. After deletion, what was cell 5 (CoordDataset) becomes cell 4.

- [ ] **Step 4: Replace the CoordDataset cell with the simplified version**

Replace the full content of the `CoordDataset` cell with:

```python
class CoordDataset(Dataset):
    """
    Dataset for TCN skeleton coordinates.

    - npy_root=None : loads from POSES_ROOT via load_coords (used for test set)
    - npy_root=Path : loads pre-computed (225, T) arrays from augmented_tcn/npy/ (used for train)
    """

    def __init__(self, df: pd.DataFrame, npy_root=None, target_T: int = TARGET_T):
        self.df       = df.reset_index(drop=True)
        self.npy_root = npy_root
        self.target_T = target_T

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.npy_root is not None:
            coords = np.load(self.npy_root / f"{row['id']}.npy").astype(np.float32)
        else:
            coords = load_coords(row["id"], self.target_T)
        return torch.from_numpy(coords), torch.tensor(row["label"], dtype=torch.long)


# Load augmented train metadata
df_aug = pd.read_csv(AUG_ROOT / "metadata.csv")
print(f"Augmented train: {len(df_aug)} instances")
print(df_aug.groupby("sign").size().reindex(SIGNS_TARGET).to_string())

train_dataset = CoordDataset(df_aug,  npy_root=AUG_ROOT / "npy")
test_dataset  = CoordDataset(df_test, npy_root=None)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=0)

xb, yb = next(iter(train_loader))
assert xb.shape == (16, 225, TARGET_T), f"Expected (16, 225, {TARGET_T}), got {xb.shape}"
print(f"\nBatch: X={xb.shape}, y={yb.shape}, dtype={xb.dtype}")
```

- [ ] **Step 5: Run the modified cell and verify batch shape assertion passes**

Expected output:
```
Augmented train: 25820 instances   ← (20 classes × 1291)
...
Batch: X=torch.Size([16, 225, 32]), y=torch.Size([16]), dtype=torch.float32
```

- [ ] **Step 6: Commit**

```bash
git add src/notebooks/02_tcn_training.ipynb
git commit -m "refactor: remove built-in augmentation from TCN training, load from augmented_tcn/"
```

---

## Self-Review

**Spec coverage check:**
- ✅ 6 augmentation functions from papers (horizontal flip, temporal flip, translation, scaling, rotation, speed)
- ✅ Output format: individual `.npy` files (225, 32) + metadata.csv
- ✅ Folder: `augmented_tcn/` (not `poses/`)
- ✅ Balancing strategy: all classes to 1291, max 50 variants/instance, repeat with new random params
- ✅ Test set unchanged
- ✅ Augmentation removed from `02_tcn_training.ipynb`
- ✅ `WeightedRandomSampler` removed
- ✅ Train loads from `augmented_tcn/npy/`, test loads from `poses/`

**Placeholder scan:** None found.

**Type consistency:**
- `load_raw_poses` → `(body, lhand, rhand)` tuple of `(T, K, 3)` arrays — used consistently in Tasks 3, 4, 5
- `apply_combo(body, lhand, rhand, combo)` → `(body, lhand, rhand)` — consistent in Tasks 4, 5
- `poses_to_coords(body, lhand, rhand)` → `(225, TARGET_T)` — consistent in Tasks 4, 5
- `CoordDataset(df, npy_root=None)` — consistent between Task 7 definition and usage
