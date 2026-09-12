"""Logging, reproducibility, serialization and metric helpers."""

import json
import logging
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

LOGGER_NAME = "vision_diagnosis"
SUBDIRS = (
    "logs", "checkpoints", "metrics", "predictions", "plots", "features", "artifacts", "splits",
)


def ensure_output_dirs(output_root):
    """Create the output tree and return a mapping of its subdirectories."""
    dirs = {"root": Path(output_root)}
    for name in SUBDIRS:
        path = Path(output_root) / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def setup_logging(log_path):
    """Configure the shared logger to write to stdout and to ``log_path``."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger():
    """Return the shared logger."""
    return logging.getLogger(LOGGER_NAME)


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_amp():
    """Return the compute device and whether float16 autocast should be used."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = use_amp
    return device, use_amp


def autocast_context(device, use_amp):
    """Return a float16 autocast context for the given device."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)


def json_default(value):
    """Convert NumPy, path and device objects into JSON-serializable values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    raise TypeError(f"Unserializable type: {type(value)}")


def save_json(data, path):
    """Write ``data`` to ``path`` as indented UTF-8 JSON."""
    path = Path(path)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    return path


def load_json(path):
    """Read JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def elapsed_hours(start):
    """Return the number of hours since ``start`` (a ``time.time()`` value)."""
    return (time.time() - start) / 3600


def environment_info(device):
    """Collect library versions and device details for the run log."""
    import open_clip
    import pandas
    import sklearn
    import torchvision

    info = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "open_clip": open_clip.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 1024 ** 3, 2),
            "gpu_count": torch.cuda.device_count(),
        })
    return info


def classification_metrics(logits, labels, num_classes):
    """Return accuracy, F1 scores, balanced accuracy and NLL for logits."""
    logits = logits.float()
    y_true = labels.numpy()
    y_pred = logits.argmax(1).numpy()
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=range(num_classes), average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=range(num_classes),
                                      average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "nll": float(F.cross_entropy(logits, labels.long()).item()),
    }


def is_better(metrics, best, min_delta):
    """Return whether ``metrics`` improve on ``best`` (macro-F1, then NLL)."""
    if metrics["macro_f1"] > best["macro_f1"] + min_delta:
        return True
    return metrics["macro_f1"] >= best["macro_f1"] and metrics["nll"] < best["nll"]


def compute_class_weights(class_indices, num_classes):
    """Return square-root inverse frequency class weights with mean one."""
    counts = np.bincount(class_indices, minlength=num_classes).astype(np.float64)
    weights = np.sqrt(counts.max() / np.maximum(counts, 1.0))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def calibration_metrics(probs, labels, n_bins):
    """Return ECE variants, NLL, Brier score and reliability-diagram bins."""
    confidence = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    count = len(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        size = int(mask.sum())
        entry = {"lower": float(edges[index]), "upper": float(edges[index + 1]), "count": size,
                 "accuracy": None, "confidence": None}
        if size:
            accuracy = float(correct[mask].mean())
            mean_confidence = float(confidence[mask].mean())
            gap = abs(accuracy - mean_confidence)
            ece += gap * size / count
            mce = max(mce, gap)
            entry.update({"accuracy": accuracy, "confidence": mean_confidence})
        bins.append(entry)
    adaptive_ece = sum(
        abs(correct[chunk].mean() - confidence[chunk].mean()) * len(chunk) / count
        for chunk in np.array_split(np.argsort(confidence), n_bins) if len(chunk)
    )
    classwise = []
    for class_index in range(probs.shape[1]):
        class_probs = probs[:, class_index]
        targets = (labels == class_index).astype(np.float64)
        ids = np.clip(np.digitize(class_probs, edges[1:-1], right=True), 0, n_bins - 1)
        classwise.append(sum(
            abs(targets[ids == index].mean() - class_probs[ids == index].mean())
            * (ids == index).sum() / count
            for index in range(n_bins) if (ids == index).any()
        ))
    onehot = np.eye(probs.shape[1])[labels]
    return {
        "accuracy": float(correct.mean()),
        "avg_confidence": float(confidence.mean()),
        "ece": float(ece),
        "mce": float(mce),
        "adaptive_ece": float(adaptive_ece),
        "classwise_ece": float(np.mean(classwise)),
        "nll": float(-np.log(np.clip(probs[np.arange(count), labels], 1e-12, 1.0)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "bins": bins,
    }


def risk_coverage(confidence, correct):
    """Return selective-risk statistics plus the coverage and risk curves."""
    count = len(correct)
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    coverage = np.arange(1, count + 1) / count
    risks = np.cumsum(errors) / np.arange(1, count + 1)
    optimal_risks = np.cumsum(np.sort(1.0 - correct)) / np.arange(1, count + 1)
    summary = {
        "aurc": float(risks.mean()),
        "eaurc": float(risks.mean() - optimal_risks.mean()),
        "overall_risk": float(errors.mean()),
    }
    for target in (0.5, 0.7, 0.9):
        summary[f"risk_at_coverage_{target}"] = float(risks[max(0, int(np.ceil(target * count)) - 1)])
    return summary, coverage, risks


def markdown_table(frame):
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join(
            f"{value:.4f}" if isinstance(value, float) else str(value) for value in row
        ) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])
