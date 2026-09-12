"""Download the raw Paddy Doctor and PlantVillage releases from Hugging Face.

Files land under ``dataset/_raw`` by default. Pass ``--raw-dir`` to change the
destination or ``--only`` to fetch a single dataset. Run ``prepare_datasets.py``
afterwards to convert them into the format the training code expects.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

SOURCES = {
    "paddy_doctor": {
        "repo_id": "Project-AgML/paddy_disease_classification",
        "allow_patterns": ["README.md", "data/*.parquet"],
    },
    "plantvillage": {
        "repo_id": "mohanty/PlantVillage",
        "allow_patterns": [
            "README.md",
            "data.zip",
            "leaf_grouping/leaf-map.json",
            "splits/color_*.txt",
        ],
    },
}

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "dataset" / "_raw"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download raw Paddy Doctor and PlantVillage files from Hugging Face."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--only", nargs="*", choices=sorted(SOURCES))
    return parser.parse_args()


def main():
    args = parse_args()
    for name in args.only or SOURCES:
        source = SOURCES[name]
        target = args.raw_dir / f"{name}_hf"
        path = snapshot_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            local_dir=target,
            allow_patterns=source["allow_patterns"],
        )
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
