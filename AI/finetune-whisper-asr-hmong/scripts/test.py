"""Evaluate a fine-tuned checkpoint on the internal (in-domain) test split."""

import json
import os

import pytorch_lightning as pl
import torch

from config import Config
from data import HmongASRDataModule
from train import WhisperLightningModule
from utils import seed_all


def run_test() -> None:
    """Load Config.CKPT_PATH and report WER/CER/RTF on the internal test split."""
    seed_all(Config.SEED)

    datamodule = HmongASRDataModule()

    # if Config.CKPT_PATH:
    #     model = WhisperLightningModule.load_from_checkpoint(Config.CKPT_PATH)
    # else:
    #     model = WhisperLightningModule()

    model = WhisperLightningModule.load_from_checkpoint(Config.CKPT_PATH)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision=Config.PRECISION,
        logger=False,
    )

    results = trainer.test(model=model, datamodule=datamodule)

    if results:
        metrics = results[0]
        wer = metrics.get("test_wer", 0.0)
        cer = metrics.get("test_cer", 0.0)
        rtf = metrics.get("test_rtf", 0.0)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(Config.OUTPUT_DIR, "test_metrics.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump({"wer": float(wer), "cer": float(cer), "rtf": float(rtf)}, file, ensure_ascii=False, indent=2)
        print(f"Average Test WER: {wer:.6f}")
        print(f"Average Test CER: {cer:.6f}")
        print(f"Average Test RTF: {rtf:.6f}")
        print(f"Saved metrics JSON: {output_path}")


if __name__ == "__main__":
    run_test()
