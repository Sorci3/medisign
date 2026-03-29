# MediSign Assist — I3D Sign Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a skeleton-based I3D model to recognize 20 LSF medical emergency signs from pose keypoint data, delivering two Jupyter notebooks: one for preprocessing and one for model training/evaluation.

**Architecture:** Pose keypoints (body, hands) for each frame are rendered into 3-channel 2D Gaussian heatmaps, then stacked into a 5D tensor `(batch, 3, T=32, H=64, W=64)` fed to an Inflated 3D Convolution Network (I3D / Inception-I3D). The small dataset (991 instances) requires strong augmentation and class weighting to combat severe class imbalance (FORT=486 vs JAMBES=1).

**Tech Stack:** Python 3.x, PyTorch, torchvision, NumPy, pandas, scikit-learn, matplotlib, seaborn, Jupyter

---

## Dataset Facts (read before touching any code)

| Property | Value |
|---|---|
| Location | `src/dataset/` |
| Total instances | 991 (with pose files) |
| Classes | 20 medical signs |
| Pose subdirs | `poses/face/` (478 kpts), `poses/left_hand/` (21 kpts), `poses/right_hand/` (21 kpts), `poses/pose/` (33 kpts) |
| Array shape | `(T, keypoints, 3)` — float16, coords already in ~[0,1] |
| Instances with T=0 | 4 (must be excluded) |
| Frame length stats | min=1, max=243, mean≈22, median=19, p90=37 |
| Pre-defined splits | `src/dataset/metadata/splits/train.json` (611), `test.json` (380) |
| Class distribution | FORT=486, AIDER=170, ... JAMBES=1 (very imbalanced) |
| Metadata | `src/dataset/instances.csv` — columns: id, sign, signer, start, end |
| Sign→index map | `src/dataset/metadata/sign_to_index.csv` |

---

## File Structure

```
notebooks/
  01_preprocessing.ipynb      # Data exploration, cleaning, heatmap generation, augmentation, save splits
  02_i3d_training.ipynb       # Dataset class, I3D architecture, training loop, metrics, model saving
src/
  dataset/                    # (existing) raw pose .npy files + CSVs
requirements.txt              # Add: torch torchvision scikit-learn matplotlib seaborn jupyter tqdm
```

---

## Task 1: Install Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add ML dependencies to requirements.txt**

Open `requirements.txt` and replace its contents with:

```
pandas
requests
lsfb_dataset
pip-system-certs
torch
torchvision
scikit-learn
matplotlib
seaborn
jupyter
tqdm
numpy
```

- [ ] **Step 2: Install them**

```bash
.venv/Scripts/pip install torch torchvision scikit-learn matplotlib seaborn jupyter tqdm numpy
```

Expected: all packages install without error. Verify with:

```bash
.venv/Scripts/python -c "import torch; import sklearn; import matplotlib; print('OK', torch.__version__)"
```

---

## Task 2: Notebook 1 — Data Exploration Cell

**Files:**
- Create: `notebooks/01_preprocessing.ipynb`

This notebook is written cell by cell. Each step below corresponds to one notebook cell.

- [ ] **Step 1: Create the notebook with a setup cell**

Create `notebooks/01_preprocessing.ipynb`. Add as **Cell 1** (code):

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from pathlib import Path

# Paths
DATASET_ROOT = Path("../src/dataset")
POSES_ROOT   = DATASET_ROOT / "poses"
METADATA_DIR = DATASET_ROOT / "metadata"

SIGNS_TARGET = [
    "SOUFFRIR", "AIDER",    "FORT",     "MALADE",   "COEUR",
    "TETE",     "MORT",     "DOS",      "VENTRE",   "FROID",
    "JAMBE",    "JAMBES",   "RESPIRER", "ACCIDENT", "FAIBLE",
    "ENCEINTE", "DIABETE",  "BRAS",     "DOSSIER",  "EFFORT"
]
SIGN_TO_IDX = {s: i for i, s in enumerate(SIGNS_TARGET)}
NUM_CLASSES  = len(SIGNS_TARGET)  # 20

print("Target classes:", NUM_CLASSES)
print("Signs:", SIGNS_TARGET)
```

- [ ] **Step 2: Add exploration cell — load instances and class distribution**

Add as **Cell 2** (code):

```python
# Load instances that have pose files
instances = pd.read_csv(DATASET_ROOT / "instances.csv")

face_files   = list((POSES_ROOT / "face").glob("*.npy"))
available_ids = {f.stem for f in face_files}

df = instances[instances["id"].isin(available_ids)].copy()
df["label"] = df["sign"].map(SIGN_TO_IDX)
df = df[df["label"].notna()].copy()  # keep only our 20 target signs
df["label"] = df["label"].astype(int)

print(f"Total instances with poses: {len(df)}")
print("\nClass distribution:")
print(df["sign"].value_counts().to_string())

# Plot class distribution
fig, ax = plt.subplots(figsize=(12, 4))
df["sign"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Instance count per sign")
ax.set_xlabel("Sign")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Add exploration cell — frame length distribution**

Add as **Cell 3** (code):

```python
# Load frame lengths for all instances
frame_lengths = {}
for _, row in df.iterrows():
    arr = np.load(POSES_ROOT / "face" / f"{row['id']}.npy")
    frame_lengths[row["id"]] = arr.shape[0]

df["T"] = df["id"].map(frame_lengths)

print("Frame length stats:")
print(df["T"].describe())
print(f"\nInstances with T=0: {(df['T'] == 0).sum()}")

fig, ax = plt.subplots(figsize=(10, 4))
df["T"].clip(upper=100).hist(bins=40, ax=ax, color="coral")
ax.axvline(32, color="red", linestyle="--", label="target T=32")
ax.set_title("Frame length distribution (clipped at 100)")
ax.set_xlabel("Frames")
ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Add exploration cell — visualize skeleton for one instance**

Add as **Cell 4** (code):

```python
def visualize_skeleton_frame(instance_id, frame_idx=0):
    """Plot body + hands keypoints for a single frame."""
    body  = np.load(POSES_ROOT / "pose"       / f"{instance_id}.npy")  # (T, 33, 3)
    lhand = np.load(POSES_ROOT / "left_hand"  / f"{instance_id}.npy")  # (T, 21, 3)
    rhand = np.load(POSES_ROOT / "right_hand" / f"{instance_id}.npy")  # (T, 21, 3)

    fig, ax = plt.subplots(figsize=(5, 7))
    for kpts, color, label in [
        (body[frame_idx],  "blue",   "body"),
        (lhand[frame_idx], "green",  "left hand"),
        (rhand[frame_idx], "orange", "right hand"),
    ]:
        ax.scatter(kpts[:, 0], 1 - kpts[:, 1], s=20, c=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{instance_id} — frame {frame_idx}")
    ax.legend()
    plt.show()

# Pick one instance per class
sample_id = df[df["sign"] == "SOUFFRIR"]["id"].iloc[0]
visualize_skeleton_frame(sample_id, frame_idx=5)
```

---

## Task 3: Notebook 1 — Preprocessing Functions

- [ ] **Step 1: Add cell — remove bad instances (T=0)**

Add as **Cell 5** (code):

```python
# Remove instances with T=0 (no pose data)
bad = df[df["T"] == 0]
print(f"Removing {len(bad)} instances with T=0:")
print(bad[["id", "sign"]].to_string())

df_clean = df[df["T"] > 0].reset_index(drop=True)
print(f"\nInstances after cleaning: {len(df_clean)}")
```

- [ ] **Step 2: Add cell — temporal resampling function**

Add as **Cell 6** (code):

```python
def resample_sequence(arr: np.ndarray, target_T: int) -> np.ndarray:
    """
    Resample a (T, K, 3) pose array to exactly target_T frames via linear interpolation.
    Works for any T >= 1.
    """
    T = arr.shape[0]
    if T == target_T:
        return arr
    src_idx = np.linspace(0, T - 1, target_T)
    lo = np.floor(src_idx).astype(int).clip(0, T - 1)
    hi = np.ceil(src_idx).astype(int).clip(0, T - 1)
    alpha = (src_idx - lo)[:, None, None]  # (target_T, 1, 1)
    return ((1 - alpha) * arr[lo] + alpha * arr[hi]).astype(arr.dtype)


# Sanity check
dummy = np.random.rand(10, 21, 3).astype(np.float16)
resampled = resample_sequence(dummy, 32)
assert resampled.shape == (32, 21, 3), f"Expected (32,21,3), got {resampled.shape}"
print("resample_sequence OK")
```

- [ ] **Step 3: Add cell — heatmap generation function**

Add as **Cell 7** (code):

```python
def keypoints_to_heatmap(kpts: np.ndarray, H: int = 64, W: int = 64,
                          sigma: float = 2.0) -> np.ndarray:
    """
    Convert (K, 3) keypoints array (x, y, conf) to a single (H, W) heatmap.
    x and y are assumed to be in [0, 1]. Confidence values < 0 are treated as
    invisible and the keypoint is skipped.

    Returns float32 array of shape (H, W) with values in [0, 1].
    """
    heatmap = np.zeros((H, W), dtype=np.float32)
    for (x, y, c) in kpts:
        if c < 0:  # invisible keypoint
            continue
        px = int(np.clip(x * W, 0, W - 1))
        py = int(np.clip(y * H, 0, H - 1))
        # Place Gaussian blob via meshgrid (vectorised)
        xs = np.arange(W)
        ys = np.arange(H)
        xx, yy = np.meshgrid(xs, ys)
        blob = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, blob)
    return heatmap


def instance_to_tensor(instance_id: str, target_T: int = 32,
                        H: int = 64, W: int = 64) -> np.ndarray:
    """
    Load the 4 pose arrays for one instance and return a (3, T, H, W) float32 tensor.

    Channels:
      0 — body pose  (33 keypoints)
      1 — left hand  (21 keypoints)
      2 — right hand (21 keypoints)
    Face is omitted (too many landmarks, low relevance for medical signs).
    """
    body  = np.load(POSES_ROOT / "pose"       / f"{instance_id}.npy").astype(np.float32)
    lhand = np.load(POSES_ROOT / "left_hand"  / f"{instance_id}.npy").astype(np.float32)
    rhand = np.load(POSES_ROOT / "right_hand" / f"{instance_id}.npy").astype(np.float32)

    body  = resample_sequence(body,  target_T)   # (T, 33, 3)
    lhand = resample_sequence(lhand, target_T)   # (T, 21, 3)
    rhand = resample_sequence(rhand, target_T)   # (T, 21, 3)

    frames = []
    for t in range(target_T):
        ch_body  = keypoints_to_heatmap(body[t],  H, W)
        ch_lhand = keypoints_to_heatmap(lhand[t], H, W)
        ch_rhand = keypoints_to_heatmap(rhand[t], H, W)
        frames.append(np.stack([ch_body, ch_lhand, ch_rhand], axis=0))  # (3, H, W)

    return np.stack(frames, axis=1)  # (3, T, H, W)


# Sanity check
sample_id = df_clean["id"].iloc[0]
t = instance_to_tensor(sample_id)
assert t.shape == (3, 32, 64, 64), f"Expected (3,32,64,64), got {t.shape}"
assert t.min() >= 0.0 and t.max() <= 1.0, "Values out of [0,1]"
print(f"instance_to_tensor OK: shape={t.shape}, min={t.min():.3f}, max={t.max():.3f}")
```

- [ ] **Step 4: Add cell — augmentation functions**

Add as **Cell 8** (code):

```python
def augment_temporal_flip(kpts_body, kpts_lhand, kpts_rhand):
    """Reverse the temporal order of the sequence."""
    return kpts_body[::-1], kpts_lhand[::-1], kpts_rhand[::-1]


def augment_horizontal_flip(kpts_body, kpts_lhand, kpts_rhand):
    """
    Flip x-coordinate (mirror left/right). Also swap left and right hand channels.
    Operates on (T, K, 3) arrays.
    """
    def flip_x(arr):
        arr = arr.copy()
        arr[:, :, 0] = 1.0 - arr[:, :, 0]
        return arr
    return flip_x(kpts_body), flip_x(kpts_rhand), flip_x(kpts_lhand)  # swap L/R


def augment_jitter(kpts_body, kpts_lhand, kpts_rhand, std: float = 0.01):
    """Add small Gaussian noise to x,y coordinates."""
    noise = np.random.randn(*kpts_body.shape).astype(np.float32) * std
    noise[:, :, 2] = 0  # do not jitter confidence
    body_j = np.clip(kpts_body + noise, 0.0, 1.0)
    noise = np.random.randn(*kpts_lhand.shape).astype(np.float32) * std
    noise[:, :, 2] = 0
    lhand_j = np.clip(kpts_lhand + noise, 0.0, 1.0)
    noise = np.random.randn(*kpts_rhand.shape).astype(np.float32) * std
    noise[:, :, 2] = 0
    rhand_j = np.clip(kpts_rhand + noise, 0.0, 1.0)
    return body_j, lhand_j, rhand_j


def augment_temporal_crop(kpts_body, kpts_lhand, kpts_rhand, target_T: int = 32):
    """
    Randomly crop a sub-sequence of target_T frames from a longer sequence.
    If T <= target_T, the sequence is left unchanged (resampling handles padding).
    """
    T = kpts_body.shape[0]
    if T <= target_T:
        return kpts_body, kpts_lhand, kpts_rhand
    start = np.random.randint(0, T - target_T)
    return (kpts_body[start:start+target_T],
            kpts_lhand[start:start+target_T],
            kpts_rhand[start:start+target_T])


print("Augmentation functions defined.")
```

---

## Task 4: Notebook 1 — Build and Save Preprocessed Dataset

- [ ] **Step 1: Add cell — build full dataset with augmentation**

Add as **Cell 9** (code):

```python
from tqdm.notebook import tqdm

TARGET_T = 32
H, W     = 64, 64

# Load the official train/test split
with open(METADATA_DIR / "splits" / "train.json") as f:
    train_ids = set(json.load(f))
with open(METADATA_DIR / "splits" / "test.json") as f:
    test_ids = set(json.load(f))

df_train = df_clean[df_clean["id"].isin(train_ids)].reset_index(drop=True)
df_test  = df_clean[df_clean["id"].isin(test_ids)].reset_index(drop=True)

print(f"Train: {len(df_train)} instances | Test: {len(df_test)} instances")
print("\nTrain class distribution:")
print(df_train["sign"].value_counts().to_string())
```

- [ ] **Step 2: Add cell — generate training tensors with augmentation**

Add as **Cell 10** (code):

```python
def load_raw_poses(instance_id: str):
    """Return (body, lhand, rhand) as float32 (T, K, 3) arrays."""
    body  = np.load(POSES_ROOT / "pose"       / f"{instance_id}.npy").astype(np.float32)
    lhand = np.load(POSES_ROOT / "left_hand"  / f"{instance_id}.npy").astype(np.float32)
    rhand = np.load(POSES_ROOT / "right_hand" / f"{instance_id}.npy").astype(np.float32)
    return body, lhand, rhand


def poses_to_tensor(body, lhand, rhand, target_T=32, H=64, W=64) -> np.ndarray:
    """Convert resampled (T, K, 3) arrays to (3, T, H, W) heatmap tensor."""
    body  = resample_sequence(body,  target_T)
    lhand = resample_sequence(lhand, target_T)
    rhand = resample_sequence(rhand, target_T)
    frames = []
    for t in range(target_T):
        frames.append(np.stack([
            keypoints_to_heatmap(body[t],  H, W),
            keypoints_to_heatmap(lhand[t], H, W),
            keypoints_to_heatmap(rhand[t], H, W),
        ], axis=0))
    return np.stack(frames, axis=1)  # (3, T, H, W)


# Build training set with augmentation (original + 3 augmented copies)
X_train, y_train = [], []

for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Building train"):
    body, lhand, rhand = load_raw_poses(row["id"])

    # Original
    X_train.append(poses_to_tensor(body, lhand, rhand, TARGET_T, H, W))
    y_train.append(row["label"])

    # Augmentation 1: horizontal flip
    b_f, l_f, r_f = augment_horizontal_flip(body, lhand, rhand)
    X_train.append(poses_to_tensor(b_f, l_f, r_f, TARGET_T, H, W))
    y_train.append(row["label"])

    # Augmentation 2: temporal flip
    b_t, l_t, r_t = augment_temporal_flip(body, lhand, rhand)
    X_train.append(poses_to_tensor(b_t, l_t, r_t, TARGET_T, H, W))
    y_train.append(row["label"])

    # Augmentation 3: spatial jitter
    b_j, l_j, r_j = augment_jitter(body, lhand, rhand, std=0.01)
    X_train.append(poses_to_tensor(b_j, l_j, r_j, TARGET_T, H, W))
    y_train.append(row["label"])

X_train = np.stack(X_train).astype(np.float32)  # (N*4, 3, T, H, W)
y_train = np.array(y_train, dtype=np.int64)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}, unique labels: {np.unique(y_train)}")
```

- [ ] **Step 3: Add cell — generate test tensors (no augmentation)**

Add as **Cell 11** (code):

```python
X_test, y_test = [], []

for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Building test"):
    body, lhand, rhand = load_raw_poses(row["id"])
    X_test.append(poses_to_tensor(body, lhand, rhand, TARGET_T, H, W))
    y_test.append(row["label"])

X_test = np.stack(X_test).astype(np.float32)   # (N_test, 3, T, H, W)
y_test = np.array(y_test, dtype=np.int64)
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")
```

- [ ] **Step 4: Add cell — save preprocessed data**

Add as **Cell 12** (code):

```python
import os

save_dir = Path("../src/dataset/preprocessed")
save_dir.mkdir(parents=True, exist_ok=True)

np.save(save_dir / "X_train.npy", X_train)
np.save(save_dir / "y_train.npy", y_train)
np.save(save_dir / "X_test.npy",  X_test)
np.save(save_dir / "y_test.npy",  y_test)

# Also save the label map for reference
import json
with open(save_dir / "sign_to_idx.json", "w") as f:
    json.dump(SIGN_TO_IDX, f, indent=2)

print("Saved preprocessed data:")
print(f"  X_train: {X_train.shape}  ({X_train.nbytes / 1e6:.1f} MB)")
print(f"  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}  ({X_test.nbytes / 1e6:.1f} MB)")
print(f"  y_test:  {y_test.shape}")
print(f"  sign_to_idx.json")
```

- [ ] **Step 5: Run Notebook 1 end-to-end and verify output files exist**

```bash
cd notebooks && ../.venv/Scripts/jupyter nbconvert --to notebook --execute 01_preprocessing.ipynb --output 01_preprocessing_executed.ipynb --ExecutePreprocessor.timeout=600
```

Expected: execution completes, files appear at `src/dataset/preprocessed/X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, `sign_to_idx.json`.

---

## Task 5: Notebook 2 — Dataset Class and DataLoader

**Files:**
- Create: `notebooks/02_i3d_training.ipynb`

- [ ] **Step 1: Add setup cell**

Create `notebooks/02_i3d_training.ipynb`. Add as **Cell 1** (code):

```python
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from pathlib import Path
from tqdm.notebook import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = Path("../src/dataset/preprocessed")

SIGNS_TARGET = [
    "SOUFFRIR", "AIDER",    "FORT",     "MALADE",   "COEUR",
    "TETE",     "MORT",     "DOS",      "VENTRE",   "FROID",
    "JAMBE",    "JAMBES",   "RESPIRER", "ACCIDENT", "FAIBLE",
    "ENCEINTE", "DIABETE",  "BRAS",     "DOSSIER",  "EFFORT"
]
NUM_CLASSES = len(SIGNS_TARGET)
IDX_TO_SIGN = {i: s for i, s in enumerate(SIGNS_TARGET)}
```

- [ ] **Step 2: Add cell — dataset class**

Add as **Cell 2** (code):

```python
class PoseHeatmapDataset(Dataset):
    """
    Wraps the preprocessed heatmap tensors.
    X shape: (N, 3, T, H, W)  float32
    y shape: (N,)              int64
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, 3, T, H, W)
        self.y = torch.from_numpy(y)  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Load data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")

print(f"Train: {X_train.shape}, labels: {y_train.shape}")
print(f"Test:  {X_test.shape},  labels: {y_test.shape}")

train_dataset = PoseHeatmapDataset(X_train, y_train)
test_dataset  = PoseHeatmapDataset(X_test,  y_test)
```

- [ ] **Step 3: Add cell — weighted sampler for class imbalance**

Add as **Cell 3** (code):

```python
# Build a WeightedRandomSampler so rare classes are sampled more often
class_counts   = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
# Multiply by 4 because training set was augmented 4×
class_weights  = 1.0 / np.maximum(class_counts, 1.0)
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights, dtype=torch.float32),
    num_samples = len(y_train),
    replacement = True,
)

train_loader = DataLoader(train_dataset, batch_size=8,  sampler=sampler,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=8,  shuffle=False,    num_workers=0)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# Verify one batch
xb, yb = next(iter(train_loader))
print(f"Batch X: {xb.shape}, y: {yb.shape}, dtype: {xb.dtype}")
```

---

## Task 6: Notebook 2 — I3D Architecture

- [ ] **Step 1: Add cell — Unit3D building block**

Add as **Cell 4** (code):

```python
class Unit3D(nn.Module):
    """3D conv + BN + ReLU. Applies 'same' padding automatically."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0, activation=True, use_bn=True):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False,
        )
        self.bn   = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

- [ ] **Step 2: Add cell — Inception module**

Add as **Cell 5** (code):

```python
class InceptionModule(nn.Module):
    """
    I3D Inception module with 4 branches:
      Branch 0: 1×1×1 conv
      Branch 1: 1×1×1 → 3×3×3 conv
      Branch 2: 1×1×1 → 3×3×3 conv  (second path)
      Branch 3: 3×3×3 max pool → 1×1×1 conv
    """

    def __init__(self, in_channels: int,
                 out_b0: int,
                 out_b1_reduce: int, out_b1: int,
                 out_b2_reduce: int, out_b2: int,
                 out_b3: int):
        super().__init__()
        self.branch0 = Unit3D(in_channels, out_b0, kernel_size=(1, 1, 1))

        self.branch1 = nn.Sequential(
            Unit3D(in_channels, out_b1_reduce, kernel_size=(1, 1, 1)),
            Unit3D(out_b1_reduce, out_b1,      kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.branch2 = nn.Sequential(
            Unit3D(in_channels, out_b2_reduce, kernel_size=(1, 1, 1)),
            Unit3D(out_b2_reduce, out_b2,      kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            Unit3D(in_channels, out_b3, kernel_size=(1, 1, 1)),
        )

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b0, b1, b2, b3], dim=1)
```

- [ ] **Step 3: Add cell — full I3D model**

Add as **Cell 6** (code):

```python
class InceptionI3D(nn.Module):
    """
    Inflated 3D Convolution Network (I3D) adapted for skeleton heatmap input.

    Input:  (batch, 3, T=32, H=64, W=64)
    Output: (batch, num_classes)

    Architecture follows Carreira & Zisserman (2017), Inception-v1 backbone inflated
    to 3D. Spatial pooling is adapted for 64×64 input (vs 224×224 in the original).
    """

    def __init__(self, num_classes: int = 20, in_channels: int = 3,
                 dropout_prob: float = 0.5):
        super().__init__()

        # --- Stem ---
        self.conv3d_1a = Unit3D(in_channels, 64, kernel_size=(7, 7, 7),
                                stride=(2, 2, 2), padding=(3, 3, 3))
        self.maxpool_2a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                       padding=(0, 1, 1))
        self.conv3d_2b = Unit3D(64, 64,  kernel_size=(1, 1, 1))
        self.conv3d_2c = Unit3D(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.maxpool_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                       padding=(0, 1, 1))

        # --- Mixed (Inception) blocks ---
        self.mixed_3b = InceptionModule(192,  64,  96, 128, 16,  32,  32)
        self.mixed_3c = InceptionModule(256, 128, 128, 192, 32,  96,  64)
        self.maxpool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                       padding=(1, 1, 1))

        self.mixed_4b = InceptionModule(480, 192,  96, 208, 16,  48,  64)
        self.mixed_4c = InceptionModule(512, 160, 112, 224, 24,  64,  64)
        self.mixed_4d = InceptionModule(512, 128, 128, 256, 24,  64,  64)
        self.mixed_4e = InceptionModule(512, 112, 144, 288, 32,  64,  64)
        self.mixed_4f = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                       padding=(0, 0, 0))

        self.mixed_5b = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # --- Head ---
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout  = nn.Dropout(p=dropout_prob)
        self.logits   = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = self.conv3d_1a(x)
        x = self.maxpool_2a(x)
        x = self.conv3d_2b(x)
        x = self.conv3d_2c(x)
        x = self.maxpool_3a(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool_4a(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.avg_pool(x)          # (B, 1024, 1, 1, 1)
        x = x.flatten(1)              # (B, 1024)
        x = self.dropout(x)
        return self.logits(x)         # (B, num_classes)


# Instantiate and verify forward pass
model = InceptionI3D(num_classes=NUM_CLASSES, dropout_prob=0.5).to(DEVICE)
dummy = torch.randn(2, 3, 32, 64, 64).to(DEVICE)
out   = model(dummy)
assert out.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {out.shape}"
print(f"I3D output shape: {out.shape}  [OK]")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

---

## Task 7: Notebook 2 — Training Loop

- [ ] **Step 1: Add cell — loss, optimizer, scheduler**

Add as **Cell 7** (code):

```python
# Class weights for cross-entropy (inverse frequency, from original train counts)
raw_counts    = np.bincount(np.load(DATA_DIR / "y_train.npy"), minlength=NUM_CLASSES)
# raw_counts already reflects augmentation (all classes × 4)
# Divide by 4 to get original frequency for weight computation
orig_counts   = np.maximum(raw_counts / 4, 1).astype(np.float32)
ce_weights    = torch.tensor(1.0 / orig_counts, dtype=torch.float32).to(DEVICE)
ce_weights   /= ce_weights.sum()  # normalise

criterion = nn.CrossEntropyLoss(weight=ce_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, verbose=True
)

print("Loss, optimizer, and scheduler defined.")
```

- [ ] **Step 2: Add cell — training and validation functions**

Add as **Cell 8** (code):

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / total
    accuracy = correct / total
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1, np.array(all_preds), np.array(all_labels)


print("train_one_epoch and evaluate defined.")
```

- [ ] **Step 3: Add cell — full training loop with early stopping**

Add as **Cell 9** (code):

```python
NUM_EPOCHS     = 50
PATIENCE       = 10
best_val_acc   = 0.0
epochs_no_improve = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step(val_acc)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")

    # Early stopping + checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "../models/best_i3d.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nBest validation accuracy: {best_val_acc:.4f}")
```

> **Note:** The `../models/` directory must exist. Add a cell before the loop:
> ```python
> import os; os.makedirs("../models", exist_ok=True)
> ```

- [ ] **Step 4: Add cell — plot training curves**

Add as **Cell 10** (code):

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"],   label="val")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"],   label="val")
axes[1].set_title("Accuracy")
axes[1].legend()

axes[2].plot(history["val_f1"], label="val macro-F1", color="green")
axes[2].set_title("Macro F1 (val)")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## Task 8: Notebook 2 — Evaluation and Model Saving

- [ ] **Step 1: Add cell — full evaluation on test set**

Add as **Cell 11** (code):

```python
# Load best checkpoint
model.load_state_dict(torch.load("../models/best_i3d.pth", map_location=DEVICE))

_, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion, DEVICE)

print(f"Test Accuracy:   {test_acc:.4f}")
print(f"Test Macro-F1:   {test_f1:.4f}")
print()
print(classification_report(
    labels, preds,
    target_names=SIGNS_TARGET,
    zero_division=0,
))
```

- [ ] **Step 2: Add cell — confusion matrix**

Add as **Cell 12** (code):

```python
cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=SIGNS_TARGET,
    yticklabels=SIGNS_TARGET,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Test Set)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Add cell — save_model function (full model or checkpoint)**

Add as **Cell 13** (code):

```python
def save_model(model: nn.Module, path: str, metadata: dict = None):
    """
    Save the trained I3D model weights and optional metadata to disk.

    Args:
        model:    trained InceptionI3D instance
        path:     file path (e.g. '../models/i3d_final.pth')
        metadata: optional dict with training info (e.g. accuracy, epochs, sign_to_idx)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    print(f"Model saved to {path}")


def load_model(path: str, num_classes: int = 20) -> InceptionI3D:
    """
    Load a previously saved model from disk.

    Args:
        path:        path to the .pth file saved by save_model
        num_classes: number of output classes (default 20)

    Returns:
        InceptionI3D instance in eval mode
    """
    payload = torch.load(path, map_location=torch.device("cpu"))
    m = InceptionI3D(num_classes=num_classes)
    m.load_state_dict(payload["state_dict"])
    m.eval()
    metadata = payload.get("metadata", {})
    if metadata:
        print("Loaded metadata:", metadata)
    return m


# Save the best model with metadata
save_model(model, "../models/i3d_medisign_final.pth", metadata={
    "test_accuracy": round(float(test_acc), 4),
    "test_macro_f1": round(float(test_f1), 4),
    "num_classes":   NUM_CLASSES,
    "signs":         SIGNS_TARGET,
    "input_shape":   [3, 32, 64, 64],
})
```

- [ ] **Step 4: Add cell — inference example**

Add as **Cell 14** (code):

```python
def predict_sign(instance_id: str, model: nn.Module, device,
                 poses_root: Path = Path("../src/dataset/poses"),
                 target_T: int = 32, H: int = 64, W: int = 64) -> tuple[str, float]:
    """
    Predict the sign label for a single instance given its ID.

    Returns: (predicted_sign_name, confidence)
    """
    model.eval()
    body  = np.load(poses_root / "pose"       / f"{instance_id}.npy").astype(np.float32)
    lhand = np.load(poses_root / "left_hand"  / f"{instance_id}.npy").astype(np.float32)
    rhand = np.load(poses_root / "right_hand" / f"{instance_id}.npy").astype(np.float32)

    tensor = poses_to_tensor(body, lhand, rhand, target_T, H, W)
    x = torch.from_numpy(tensor).unsqueeze(0).to(device)  # (1, 3, T, H, W)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        idx    = probs.argmax(1).item()
        conf   = probs[0, idx].item()
    return IDX_TO_SIGN[idx], conf


# Test on a few instances
sample_df = pd.read_csv("../src/dataset/instances.csv")
face_files = list(Path("../src/dataset/poses/face").glob("*.npy"))
available  = {f.stem for f in face_files}
sample_df  = sample_df[sample_df["id"].isin(available)]
sample_df  = sample_df[sample_df["sign"].isin(SIGNS_TARGET)]

print("Inference examples:")
for sign in ["SOUFFRIR", "AIDER", "FORT", "COEUR"]:
    row = sample_df[sample_df["sign"] == sign].iloc[0]
    pred, conf = predict_sign(row["id"], model, DEVICE)
    status = "✓" if pred == sign else "✗"
    print(f"  {status} True: {sign:<12} Pred: {pred:<12} Conf: {conf:.3f}")
```

---

## Task 9: Final Verification

- [ ] **Step 1: Run Notebook 2 end-to-end**

```bash
cd notebooks && ../.venv/Scripts/jupyter nbconvert --to notebook --execute 02_i3d_training.ipynb --output 02_i3d_training_executed.ipynb --ExecutePreprocessor.timeout=3600
```

Expected: no exceptions, final cell prints inference examples with confidence scores.

- [ ] **Step 2: Verify saved model files exist**

```bash
ls -lh ../models/
```

Expected output (paths relative to `notebooks/`):

```
best_i3d.pth
i3d_medisign_final.pth
```

- [ ] **Step 3: Verify model can be reloaded**

```bash
../.venv/Scripts/python -c "
import torch, sys
sys.path.insert(0, 'notebooks')
payload = torch.load('../models/i3d_medisign_final.pth', map_location='cpu')
print('Metadata:', payload['metadata'])
print('State dict keys (first 5):', list(payload['state_dict'].keys())[:5])
print('Model reload: OK')
"
```

Expected: prints metadata dict with `test_accuracy`, `test_macro_f1`, and `signs` keys without error.

---

## Self-Review Notes

**Spec coverage check:**
- [x] 20 target LSF medical signs
- [x] Pose coordinates as input (not raw video)
- [x] Preprocessing notebook (01) — augmentation, cleaning, normalization
- [x] I3D model implementation
- [x] Model evaluation with accuracy and macro-F1
- [x] Confusion matrix per-class breakdown
- [x] `save_model` function that saves the trained model
- [x] `load_model` function for reuse

**Known limitations (document in notebook markdown cells):**
- Dataset is tiny (991 instances, 20 classes). JAMBES has only 1 instance — its evaluation metrics are unreliable.
- I3D was designed for large-scale video datasets. With this dataset size, performance will be modest. Transfer learning from Kinetics pretrained weights (not included here) would significantly improve results.
- Face keypoints (478 landmarks) are excluded to keep input compact and reduce overfitting.
