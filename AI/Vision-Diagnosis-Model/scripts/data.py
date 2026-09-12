"""Dataset loading, near-duplicate grouping, splitting and feature extraction."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

import config
import utils

SPLIT_COLUMNS = [
    "image_id", "source", "label", "relpath", "group_id", "split_group", "class_index",
    "class_name", "fold", "split",
]
OOD_COLUMNS = ["image_id", "source", "label", "relpath", "group_id", "ood_type", "split"]
SOURCE_DIRS = {"paddy_doctor": config.PADDY_DIR, "plantvillage": config.PLANTVILLAGE_DIR}


def resolve_dataset_root(path):
    """Find the directory holding ``metadata.csv`` under ``path``."""
    path = Path(path)
    for pattern in ("metadata.csv", "*/metadata.csv", "*/*/metadata.csv", "*/*/*/metadata.csv"):
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0].parent
    raise FileNotFoundError(f"metadata.csv not found under {path}")


def load_standard_dataset(path, source):
    """Load a prepared dataset and return its metadata with absolute file paths."""
    root = resolve_dataset_root(path)
    frame = pd.read_csv(root / "metadata.csv")
    missing = {"image_id", "relpath", "label", "group_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"{root}: missing columns {sorted(missing)}")
    frame["source"] = source
    frame["filepath"] = [str(root / relpath) for relpath in frame["relpath"]]
    frame["group_id"] = source + ":" + frame["group_id"].astype(str)
    sample = frame["filepath"].sample(min(50, len(frame)), random_state=0)
    absent = [filepath for filepath in sample if not Path(filepath).exists()]
    if absent:
        raise FileNotFoundError(f"{root}: image files missing, e.g. {absent[0]}")
    utils.get_logger().info(
        "Loaded %s from %s: %d images, %d labels", source, root, len(frame), frame["label"].nunique()
    )
    return frame


def cap_per_label(frame, cap, seed):
    """Sample at most ``cap`` images per label."""
    parts = [
        group.sample(n=min(len(group), cap), random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ]
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def build_pools(paddy, plantvillage):
    """Split both datasets into the in-distribution, near-OOD and far-OOD pools."""
    lookup = {(item["source"], item["label"]): index for index, item in enumerate(config.CLASSES)}
    combined = pd.concat([paddy, plantvillage], ignore_index=True)
    combined["class_index"] = [
        lookup.get(key, -1) for key in zip(combined["source"], combined["label"])
    ]
    missing = [
        item["name"] for index, item in enumerate(config.CLASSES)
        if not (combined["class_index"] == index).any()
    ]
    if missing:
        raise ValueError(f"Classes without images: {missing}")
    in_dist = cap_per_label(
        combined[combined["class_index"] >= 0], config.DATA["max_images_per_class"], config.SEED
    )
    in_dist["class_name"] = [config.CLASS_NAMES[index] for index in in_dist["class_index"]]
    near_mask = (combined["source"] == "paddy_doctor") & combined["label"].isin(
        config.NEAR_OOD_LABELS
    )
    far_mask = (combined["source"] == config.FAR_OOD["source"]) & ~combined["label"].str.startswith(
        config.FAR_OOD["exclude_prefix"]
    )
    near = cap_per_label(combined[near_mask], config.DATA["near_ood_per_label"], config.SEED)
    far = cap_per_label(combined[far_mask], config.DATA["far_ood_per_label"], config.SEED)
    near["ood_type"] = "near"
    far["ood_type"] = "far"
    for frame in (near, far):
        frame["class_index"] = -1
        frame["class_name"] = "ood"
    return in_dist, near, far


def perceptual_hash(filepath):
    """Return the 64-bit perceptual hash of an image as a boolean array."""
    with Image.open(filepath) as image:
        image.draft("L", (64, 64))
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = scipy.fft.dctn(np.asarray(gray, dtype=np.float64), norm="ortho")[:8, :8]
    return (coefficients > np.median(coefficients)).reshape(-1)


class UnionFind:
    """Disjoint-set forest over integer indices."""

    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, item):
        """Return the representative of the set containing ``item``."""
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, first, second):
        """Merge the sets containing ``first`` and ``second``."""
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def assign_split_groups(frame, threshold, device):
    """Group images by source group id and by perceptual-hash near duplicates."""
    start = time.time()
    with ThreadPoolExecutor() as executor:
        hashes = list(executor.map(perceptual_hash, frame["filepath"]))
    bits = torch.from_numpy(np.stack(hashes).astype(np.float32)).to(device)
    finder = UnionFind(len(frame))
    for positions in frame.groupby("group_id").indices.values():
        for position in positions[1:]:
            finder.union(positions[0], position)
    near_duplicate_pairs = 0
    for offset in range(0, len(frame), 2048):
        block = bits[offset:offset + 2048]
        distance = block @ (1 - bits).T + (1 - block) @ bits.T
        rows, cols = torch.nonzero(distance <= threshold, as_tuple=True)
        for row, col in zip((rows + offset).tolist(), cols.tolist()):
            if row < col:
                finder.union(row, col)
                near_duplicate_pairs += 1
    roots = np.array([finder.find(index) for index in range(len(frame))])
    frame = frame.assign(split_group=[f"g{root}" for root in roots])
    group_sizes = frame["split_group"].value_counts()
    stats = {
        "images": int(len(frame)),
        "source_groups": int(frame["group_id"].nunique()),
        "split_groups": int(len(group_sizes)),
        "near_duplicate_pairs": int(near_duplicate_pairs),
        "hamming_threshold": threshold,
        "largest_group": int(group_sizes.max()),
        "groups_with_multiple_images": int((group_sizes > 1).sum()),
        "seconds": round(time.time() - start, 1),
    }
    utils.get_logger().info("Grouping: %s", stats)
    return frame, stats


def assign_splits(frame):
    """Assign grouped, stratified train / val / calib / test folds."""
    splitter = StratifiedGroupKFold(
        n_splits=config.DATA["n_folds"], shuffle=True, random_state=config.SEED
    )
    folds = np.full(len(frame), -1)
    for fold, (_, index) in enumerate(
        splitter.split(np.zeros(len(frame)), frame["class_index"], frame["split_group"])
    ):
        folds[index] = fold
    fold_to_split = {
        fold: name for name in config.SPLIT_NAMES for fold in config.DATA[f"{name}_folds"]
    }
    frame = frame.assign(fold=folds, split=[fold_to_split[fold] for fold in folds])
    groups = {
        name: set(frame.loc[frame["split"] == name, "split_group"]) for name in config.SPLIT_NAMES
    }
    for position, first in enumerate(config.SPLIT_NAMES):
        for second in config.SPLIT_NAMES[position + 1:]:
            if groups[first] & groups[second]:
                raise RuntimeError(f"Group leakage between {first} and {second}")
    return frame


def split_ood_half(frame, seed):
    """Split an OOD pool into a validation and a test half by group."""
    groups = frame["group_id"].unique()
    rng = np.random.default_rng(seed)
    val_groups = set(rng.permutation(groups)[: len(groups) // 2])
    return frame.assign(split=np.where(frame["group_id"].isin(val_groups), "ood_val", "ood_test"))


def build_frames(in_dist, near_ood, far_ood):
    """Return one frame per split, indexed by the canonical split names."""
    frames = {name: in_dist[in_dist["split"] == name] for name in config.SPLIT_NAMES}
    frames["near_ood_val"] = near_ood[near_ood["split"] == "ood_val"]
    frames["near_ood_test"] = near_ood[near_ood["split"] == "ood_test"]
    frames["far_ood_val"] = far_ood[far_ood["split"] == "ood_val"]
    frames["far_ood_test"] = far_ood[far_ood["split"] == "ood_test"]
    return {name: frame.reset_index(drop=True) for name, frame in frames.items()}


def prepare_all(device):
    """Load both datasets and produce grouped, leakage-checked splits."""
    paddy = load_standard_dataset(config.PADDY_DIR, "paddy_doctor")
    plantvillage = load_standard_dataset(config.PLANTVILLAGE_DIR, "plantvillage")
    in_dist, near_ood, far_ood = build_pools(paddy, plantvillage)
    in_dist, dedup_stats = assign_split_groups(
        in_dist, config.DATA["phash_hamming_threshold"], device
    )
    in_dist = assign_splits(in_dist)
    near_ood = split_ood_half(near_ood, config.SEED)
    far_ood = split_ood_half(far_ood, config.SEED + 1)
    frames = build_frames(in_dist, near_ood, far_ood)
    utils.get_logger().info(
        "Set sizes: %s", {name: len(frame) for name, frame in frames.items()}
    )
    return frames, in_dist, near_ood, far_ood, dedup_stats


def write_split_files(dirs, in_dist, near_ood, far_ood, frames, dedup_stats):
    """Write the split tables and the grouping summary."""
    counts = pd.crosstab(in_dist["class_name"], in_dist["split"], margins=True)
    counts = counts[list(config.SPLIT_NAMES) + ["All"]]
    counts.to_csv(dirs["splits"] / "split_counts.csv")
    in_dist[SPLIT_COLUMNS].to_csv(dirs["splits"] / "in_distribution_splits.csv", index=False)
    pd.concat([near_ood, far_ood])[OOD_COLUMNS].to_csv(
        dirs["splits"] / "ood_splits.csv", index=False
    )
    utils.save_json(
        {"dedup": dedup_stats, "sizes": {name: len(frame) for name, frame in frames.items()}},
        dirs["splits"] / "split_summary.json",
    )
    utils.get_logger().info("Split counts per class:\n%s", counts.to_string())


def frames_from_splits(dirs):
    """Rebuild the split frames from the tables written during training."""
    in_dist = pd.read_csv(dirs["splits"] / "in_distribution_splits.csv")
    ood = pd.read_csv(dirs["splits"] / "ood_splits.csv")
    for frame in (in_dist, ood):
        frame["filepath"] = [
            str(SOURCE_DIRS[source] / relpath)
            for source, relpath in zip(frame["source"], frame["relpath"])
        ]
    ood["class_index"] = -1
    ood["class_name"] = "ood"
    near = ood[ood["ood_type"] == "near"]
    far = ood[ood["ood_type"] == "far"]
    return build_frames(in_dist, near, far)


def split_labels(frames):
    """Return label tensors for the in-distribution splits."""
    return {
        name: torch.tensor(frames[name]["class_index"].to_numpy()) for name in config.SPLIT_NAMES
    }


def build_transforms(image_size, mean, std):
    """Return the training and evaluation image transforms."""
    interpolation = InterpolationMode.BICUBIC
    train_ops = [
        v2.RandomResizedCrop(image_size, scale=(0.35, 1.0), interpolation=interpolation,
                             antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.25, 0.03)], p=0.8),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
    ]
    if hasattr(v2, "JPEG"):
        train_ops.append(v2.RandomApply([v2.JPEG(quality=(40, 95))], p=0.3))
    train_ops += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ]
    eval_ops = [
        v2.Resize(image_size, interpolation=interpolation, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ]
    return v2.Compose(train_ops), v2.Compose(eval_ops)


class ImageDataset(Dataset):
    """Image classification dataset backed by a metadata frame."""

    def __init__(self, frame, transform):
        self.filepaths = frame["filepath"].tolist()
        self.labels = frame["class_index"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        with Image.open(self.filepaths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index]), index


def make_loader(frame, transform, batch_size, shuffle, device, persistent=False):
    """Build a DataLoader over ``frame`` with the project data settings."""
    workers = config.DATA["num_workers"]
    return DataLoader(
        ImageDataset(frame, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(frame) >= 2 * batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent and workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        generator=torch.Generator().manual_seed(config.SEED) if shuffle else None,
    )


@torch.no_grad()
def extract_features(visual, loader, device, use_amp):
    """Return L2-normalized embeddings for every image in ``loader``."""
    visual.eval()
    outputs = []
    for images, _, _ in loader:
        with utils.autocast_context(device, use_amp):
            features = visual(images.to(device, non_blocking=True))
        outputs.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs)


def extract_all_features(visual, frames, transform, device, use_amp, tag=""):
    """Extract embeddings for every split and log the throughput."""
    logger = utils.get_logger()
    features = {}
    for name, frame in frames.items():
        start = time.time()
        loader = make_loader(
            frame, transform, config.DATA["eval_batch_size"], shuffle=False, device=device
        )
        features[name] = extract_features(visual, loader, device, use_amp)
        elapsed = time.time() - start
        logger.info(
            "[%s] features %s: %d images, %.1fs (%.1f img/s)", tag, name, len(frame), elapsed,
            len(frame) / max(elapsed, 1e-6),
        )
    return features


def save_features(path, features, frames):
    """Save embeddings together with the image ids they were computed from."""
    payload = {
        name: {"features": tensor.half(), "image_ids": frames[name]["image_id"].tolist()}
        for name, tensor in features.items()
    }
    torch.save(payload, path)


def load_features(path, frames):
    """Load embeddings and reorder ``frames`` to match the stored image ids."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = {}
    ordered = {}
    for name, entry in payload.items():
        image_ids = entry["image_ids"]
        frame = frames[name].set_index("image_id")
        unknown = set(image_ids) - set(frame.index)
        if unknown:
            raise KeyError(f"{name}: {len(unknown)} image ids in features are not in the splits")
        ordered[name] = frame.loc[image_ids].reset_index()
        features[name] = entry["features"].float()
    return features, ordered
