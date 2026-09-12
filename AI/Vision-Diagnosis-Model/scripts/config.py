"""Configuration for the Vision Diagnosis Model.

Paths are the only values that normally need editing. ``DATA_ROOT`` must contain
the two prepared datasets (``paddy_doctor`` and ``plantvillage``) and
``OUTPUT_ROOT`` receives logs, checkpoints, metrics and artifacts. Both can also
be set through the ``AGV_DATA_ROOT`` and ``AGV_OUTPUT_ROOT`` environment
variables. Setting ``AGV_SMOKE_TEST=1`` switches to a tiny configuration that
only verifies that the pipeline runs end to end on CPU.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("AGV_DATA_ROOT", PROJECT_ROOT / "dataset"))
OUTPUT_ROOT = Path(os.environ.get("AGV_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "local_run"))
PADDY_DIR = Path(os.environ.get("AGV_PADDY_DIR", DATA_ROOT / "paddy_doctor"))
PLANTVILLAGE_DIR = Path(os.environ.get("AGV_PLANTVILLAGE_DIR", DATA_ROOT / "plantvillage"))

SMOKE_TEST = os.environ.get("AGV_SMOKE_TEST", "0") == "1"
SEED = 42
TIME_BUDGET_HOURS = 10.5

CLASSES = [
    {"name": "rice__normal", "display_vi": "Lúa - Khỏe mạnh",
     "source": "paddy_doctor", "label": "normal"},
    {"name": "rice__blast", "display_vi": "Lúa - Đạo ôn",
     "source": "paddy_doctor", "label": "blast"},
    {"name": "rice__bacterial_leaf_blight", "display_vi": "Lúa - Bạc lá",
     "source": "paddy_doctor", "label": "bacterial_leaf_blight"},
    {"name": "rice__brown_spot", "display_vi": "Lúa - Đốm nâu",
     "source": "paddy_doctor", "label": "brown_spot"},
    {"name": "rice__tungro", "display_vi": "Lúa - Tungro",
     "source": "paddy_doctor", "label": "tungro"},
    {"name": "rice__hispa", "display_vi": "Lúa - Bọ gai",
     "source": "paddy_doctor", "label": "hispa"},
    {"name": "corn__healthy", "display_vi": "Ngô - Khỏe mạnh",
     "source": "plantvillage", "label": "Corn_(maize)___healthy"},
    {"name": "corn__gray_leaf_spot", "display_vi": "Ngô - Đốm lá xám",
     "source": "plantvillage", "label": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"},
    {"name": "corn__common_rust", "display_vi": "Ngô - Gỉ sắt",
     "source": "plantvillage", "label": "Corn_(maize)___Common_rust_"},
    {"name": "corn__northern_leaf_blight", "display_vi": "Ngô - Đốm lá lớn",
     "source": "plantvillage", "label": "Corn_(maize)___Northern_Leaf_Blight"},
]
CLASS_NAMES = [item["name"] for item in CLASSES]
NUM_CLASSES = len(CLASSES)

NEAR_OOD_LABELS = ["bacterial_leaf_streak", "bacterial_panicle_blight", "downy_mildew", "dead_heart"]
FAR_OOD = {"source": "plantvillage", "exclude_prefix": "Corn_(maize)___"}

MODEL = {"name": "hf-hub:imageomics/bioclip-2", "pretrained": None, "image_size": 224}

DATA = {
    "max_images_per_class": 1500,
    "near_ood_per_label": 300,
    "far_ood_per_label": 40,
    "phash_hamming_threshold": 4,
    "n_folds": 10,
    "train_folds": [0, 1, 2, 3, 4, 5, 6],
    "val_folds": [7],
    "calib_folds": [8],
    "test_folds": [9],
    "num_workers": min(4, os.cpu_count() or 1),
    "eval_batch_size": 256,
}

LINEAR_PROBE = {
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "batch_size": 1024,
    "max_epochs": 500,
    "patience": 40,
    "min_delta": 1e-4,
}

LORA = {
    "enabled": True,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "last_n_blocks": 12,
    "targets": ["c_fc", "c_proj"],
    "lr_lora": 2e-4,
    "lr_head": 5e-4,
    "weight_decay": 0.01,
    "reference_batch_size": 64,
    "batch_size_candidates": [128, 96, 64, 48, 32, 16],
    "max_memory_fraction": 0.85,
    "warmup_epochs": 1,
    "max_epochs": 40,
    "patience": 8,
    "min_delta": 1e-3,
    "grad_clip": 1.0,
    "min_lr_ratio": 0.05,
}

CALIBRATION = {"n_bins": 15}
OOD = {"tpr": 0.95, "knn_k": 50, "methods": ["msp", "energy", "knn"]}
CONFORMAL = {"alpha": 0.1, "raps_lambda": 0.01, "methods": ["thr", "aps", "raps"]}
GATE_PREVIEW = {"high_prob_threshold": 0.7, "max_medium_set_size": 3}

SPLIT_NAMES = ("train", "val", "calib", "test")
OOD_SPLIT_NAMES = ("near_ood_val", "near_ood_test", "far_ood_val", "far_ood_test")


def _apply_smoke_overrides():
    """Shrink the configuration so a full run finishes in minutes on CPU."""
    MODEL.update({"name": "ViT-B-32", "pretrained": None})
    DATA.update({
        "max_images_per_class": 12,
        "near_ood_per_label": 6,
        "far_ood_per_label": 2,
        "num_workers": 0,
        "eval_batch_size": 16,
    })
    LINEAR_PROBE.update({"max_epochs": 5, "patience": 2})
    LORA.update({
        "last_n_blocks": 2,
        "batch_size_candidates": [8],
        "max_epochs": 2,
        "patience": 1,
    })
    OOD["knn_k"] = 5


if SMOKE_TEST:
    _apply_smoke_overrides()


def as_dict():
    """Return the configuration as a JSON-serializable dictionary."""
    return {
        "seed": SEED,
        "smoke_test": SMOKE_TEST,
        "time_budget_hours": TIME_BUDGET_HOURS,
        "paths": {
            "paddy": str(PADDY_DIR),
            "plantvillage": str(PLANTVILLAGE_DIR),
            "output": str(OUTPUT_ROOT),
        },
        "classes": CLASSES,
        "near_ood_labels": NEAR_OOD_LABELS,
        "far_ood": FAR_OOD,
        "model": MODEL,
        "data": DATA,
        "linear_probe": LINEAR_PROBE,
        "lora": LORA,
        "calibration": CALIBRATION,
        "ood": OOD,
        "conformal": CONFORMAL,
        "gate_preview": GATE_PREVIEW,
    }
