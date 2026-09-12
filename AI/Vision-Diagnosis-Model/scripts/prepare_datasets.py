"""Convert the raw downloads into the shared ``images/`` plus ``metadata.csv`` format.

Reads ``dataset/_raw`` and writes ``dataset/paddy_doctor`` and
``dataset/plantvillage``. Pass ``--out-dir`` to change the destination, ``--only``
to convert a single dataset, or ``--overwrite`` to rebuild an existing one.
"""

import argparse
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "dataset" / "_raw"
DEFAULT_OUT_DIR = ROOT / "dataset"

PADDY_CLASSES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

METADATA_COLUMNS = [
    "image_id",
    "relpath",
    "label",
    "group_id",
    "source",
    "original_name",
    "width",
    "height",
]

DATASET_INFO = {
    "paddy_doctor": {
        "source_repo": "https://huggingface.co/datasets/Project-AgML/paddy_disease_classification",
        "origin": "Paddy Doctor, Kaggle competition 'Paddy Disease Classification' labeled train set",
        "license": "See original Paddy Doctor release (IEEE DataPort / Kaggle competition rules)",
        "citation": (
            "Petchiammal A, Briskline Kiruba S, Murugan D, Pandarasamy Arjunan. (2022). "
            "Paddy Doctor: A Visual Image Dataset for Automated Paddy Disease Classification "
            "and Benchmarking. IEEE Dataport. https://dx.doi.org/10.21227/hz4v-af08"
        ),
        "group_id_note": "No leaf grouping available; group_id equals image_id.",
    },
    "plantvillage": {
        "source_repo": "https://huggingface.co/datasets/mohanty/PlantVillage",
        "origin": "PlantVillage raw/color images",
        "license": "CC BY-SA 3.0",
        "citation": (
            "Mohanty, S. P., Hughes, D. P., Salathe, M. (2016). Using deep learning for "
            "image-based plant disease detection. Frontiers in Plant Science, 7. "
            "https://doi.org/10.3389/fpls.2016.01419"
        ),
        "group_id_note": "group_id is the physical leaf id from leaf_grouping/leaf-map.json.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw downloads into the shared images/<class>/ + metadata.csv format."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", nargs="*", choices=sorted(DATASET_INFO))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_extension(data):
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    raise ValueError("Unsupported image encoding")


def image_size(data):
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def prepare_output(out_root, overwrite):
    if (out_root / "metadata.csv").exists() and not overwrite:
        print(f"skip {out_root.name}: metadata.csv exists (use --overwrite)")
        return False
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    return True


def write_outputs(out_root, name, rows):
    frame = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    frame.to_csv(out_root / "metadata.csv", index=False)
    counts = Counter(frame["label"])
    info = {
        "name": name,
        **DATASET_INFO[name],
        "format": {
            "images": "images/<label>/<image_id>.<ext>",
            "metadata": "metadata.csv with columns " + ", ".join(METADATA_COLUMNS),
        },
        "num_images": int(len(frame)),
        "num_classes": len(counts),
        "num_groups": int(frame["group_id"].nunique()),
        "class_counts": dict(sorted(counts.items())),
    }
    (out_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{name}: {len(frame)} images, {len(counts)} classes -> {out_root}")


def paddy_class_names(parquet_file):
    metadata = parquet_file.schema_arrow.metadata or {}
    raw = metadata.get(b"huggingface")
    if raw:
        features = json.loads(raw).get("info", {}).get("features", {})
        names = features.get("label", {}).get("names")
        if names:
            return names
    return PADDY_CLASSES


def prepare_paddy_doctor(raw_dir, out_root):
    parquet_paths = sorted((raw_dir / "paddy_doctor_hf" / "data").glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError("Paddy Doctor parquet files not found; run download_datasets.py")
    rows = []
    index = 0
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        class_names = paddy_class_names(parquet_file)
        for batch in parquet_file.iter_batches(batch_size=256, columns=["image", "label"]):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image, label_index in zip(images, labels):
                data = image["bytes"]
                label = class_names[label_index]
                image_id = f"pd_{index:05d}"
                relpath = PurePosixPath("images", label, image_id + image_extension(data))
                target = out_root / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                width, height = image_size(data)
                rows.append(
                    [image_id, str(relpath), label, image_id, "paddy_doctor",
                     image.get("path") or "", width, height]
                )
                index += 1
    write_outputs(out_root, "paddy_doctor", rows)


def plantvillage_leaf_id(class_name, file_name, leaf_map):
    identifier = file_name.replace("_final_masked", "")
    if "___" in identifier:
        identifier = identifier.split("___")[-1]
    identifier = identifier.split("copy")[0]
    for extension in (".jpg", ".JPG", ".png", ".PNG"):
        identifier = identifier.replace(extension, "")
    identifier = identifier.strip()
    suggestions = leaf_map.get(identifier.lower().strip())
    if not suggestions:
        return f"fallback_{class_name}_{identifier}"
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return f"fallback_{class_name}_{identifier}"


def prepare_plantvillage(raw_dir, out_root):
    source_dir = raw_dir / "plantvillage_hf"
    zip_path = source_dir / "data.zip"
    leaf_map_path = source_dir / "leaf_grouping" / "leaf-map.json"
    if not zip_path.exists() or not leaf_map_path.exists():
        raise FileNotFoundError("PlantVillage data.zip or leaf-map.json missing; run download_datasets.py")
    leaf_map = json.loads(leaf_map_path.read_text(encoding="utf-8"))
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.startswith("raw/color/")
        )
        for index, member in enumerate(members):
            parts = PurePosixPath(member).parts
            if len(parts) != 4:
                continue
            class_name, file_name = parts[2], parts[3]
            data = archive.read(member)
            image_id = f"pv_{index:05d}"
            relpath = PurePosixPath("images", class_name, image_id + image_extension(data))
            target = out_root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            width, height = image_size(data)
            rows.append(
                [image_id, str(relpath), class_name,
                 plantvillage_leaf_id(class_name, file_name, leaf_map),
                 "plantvillage", member, width, height]
            )
    write_outputs(out_root, "plantvillage", rows)


PREPARERS = {
    "paddy_doctor": prepare_paddy_doctor,
    "plantvillage": prepare_plantvillage,
}


def main():
    args = parse_args()
    for name in args.only or PREPARERS:
        out_root = args.out_dir / name
        if prepare_output(out_root, args.overwrite):
            PREPARERS[name](args.raw_dir, out_root)


if __name__ == "__main__":
    main()
