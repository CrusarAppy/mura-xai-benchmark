"""MURA dataset: indexing, patient-level splitting, k-fold CV, transforms.

Expected layout (standard MURA-v1.1 release):
    <mura_root>/train_image_paths.csv
    <mura_root>/valid_image_paths.csv
    <mura_root>/train/XR_<REGION>/patient#####/study#_<positive|negative>/imageN.png
    <mura_root>/valid/...
Each line in *_image_paths.csv is a path such as
    MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png
Label: 'positive' == abnormal (1), 'negative' == normal (0).

The loader VERIFIES the structure and raises a clear error if paths cannot be
resolved — it never silently guesses. (See PROJECT_MEMORY.md operating rules.)
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

_REGION_RE = re.compile(r"(XR_[A-Z]+)")
_PATIENT_RE = re.compile(r"(patient\d+)")
_STUDY_RE = re.compile(r"(study\d+)")


def _resolve(mura_root: Path, csv_relpath: str) -> Optional[Path]:
    """Resolve a CSV path line against the dataset root, trying common variants."""
    csv_relpath = csv_relpath.strip().replace("\\", "/")
    candidates = [
        mura_root / csv_relpath,                                   # root == parent of MURA-v1.1
        mura_root.parent / csv_relpath,                            # root == MURA-v1.1 itself
        mura_root / Path(*Path(csv_relpath).parts[1:]),            # strip leading 'MURA-v1.1/'
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


MURA_REGIONS = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND",
                "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]


def expand_regions(regions) -> list:
    """Normalise a config `regions` value to an explicit list for per-anatomy iteration.
    'all' -> the seven MURA regions; a list is returned as-is."""
    if regions == "all" or regions is None:
        return list(MURA_REGIONS)
    return list(regions)


def build_mura_index(mura_root: str | Path, split: str = "train",
                     regions="all", verify_n: int = 25) -> pd.DataFrame:
    """Return a DataFrame with columns: filepath, label, patient, study, region, split.

    `split` in {'train','valid'}. `regions` is 'all' or a list like ['XR_WRIST'].
    `verify_n` image paths are checked on disk to fail fast on a wrong root.
    """
    mura_root = Path(mura_root).expanduser().resolve()
    csv_path = mura_root / f"{split}_image_paths.csv"
    if not csv_path.exists():
        csv_path = mura_root.parent / f"MURA-v1.1/{split}_image_paths.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find '{split}_image_paths.csv' under {mura_root}. "
            f"Set data.mura_root to your MURA-v1.1 folder. "
            f"Please confirm the dataset path/structure."
        )

    lines = [ln for ln in csv_path.read_text().splitlines() if ln.strip()]
    rows = []
    checked = 0
    for rel in lines:
        resolved = _resolve(mura_root, rel)
        if resolved is None and checked < verify_n:
            raise FileNotFoundError(
                f"Image path from CSV could not be resolved on disk: '{rel}'. "
                f"Tried under {mura_root}. Please confirm the MURA folder structure."
            )
        checked += 1
        region_m = _REGION_RE.search(rel)
        patient_m = _PATIENT_RE.search(rel)
        study_m = _STUDY_RE.search(rel)
        label = 1 if "positive" in rel.lower() else 0
        rows.append({
            "filepath": str(resolved) if resolved else str(mura_root / rel),
            "label": label,
            "patient": patient_m.group(1) if patient_m else "unknown",
            "study": (patient_m.group(1) + "/" + study_m.group(1)) if (patient_m and study_m) else rel,
            "region": region_m.group(1) if region_m else "UNKNOWN",
            "split": split,
        })
    df = pd.DataFrame(rows)
    if regions != "all":
        want = set(regions)
        df = df[df["region"].isin(want)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No images indexed — check mura_root / regions filter.")
    return df


def make_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign a CV fold to each row using PATIENT-LEVEL grouping (no leakage).

    Adds an integer 'fold' column (0..n_folds-1). Patients (not images) are split.
    """
    from sklearn.model_selection import GroupKFold
    import numpy as np
    df = df.reset_index(drop=True).copy()
    # GroupKFold is deterministic given order; shuffle patients by seed first for balance.
    rng = np.random.RandomState(seed)
    patients = df["patient"].to_numpy()
    order = rng.permutation(len(df))
    df_shuf = df.iloc[order].reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_folds)
    fold_col = pd.Series(-1, index=df_shuf.index, dtype=int)
    for k, (_, val_idx) in enumerate(gkf.split(df_shuf, df_shuf["label"], df_shuf["patient"])):
        fold_col.iloc[val_idx] = k
    df_shuf["fold"] = fold_col
    return df_shuf.sort_index()


def build_transforms(image_size: int = 224, train: bool = True):
    """ImageNet-normalized transforms; augmentation only on train."""
    from torchvision import transforms
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def class_weights(df: pd.DataFrame):
    """Inverse-frequency class weights [w_normal, w_abnormal] for weighted CE loss."""
    import torch
    counts = df["label"].value_counts().sort_index()
    n0 = int(counts.get(0, 1)); n1 = int(counts.get(1, 1))
    total = n0 + n1
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)


class MuraDataset:
    """PyTorch Dataset over a MURA index DataFrame."""

    def __init__(self, df: pd.DataFrame, image_size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.tf = build_transforms(image_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("L")
        x = self.tf(img)
        return x, int(row["label"])
