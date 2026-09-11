"""Evaluate a fine-tuned checkpoint on an external, out-of-domain test set."""

import json
import os
from typing import Any

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from config import Config
from data import WhisperDataCollator
from train import WhisperLightningModule
from utils import normalize_text, seed_all


def load_audio_16k(audio_path: str) -> np.ndarray | None:
    """Load, mono-mix, resample to 16 kHz, and peak-normalize an audio file.

    Returns None if the file cannot be decoded or is empty.
    """
    audio_array: np.ndarray | None = None

    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        if int(sample_rate) != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=int(sample_rate), target_sr=16000)
    except Exception:
        try:
            audio_fallback, _ = librosa.load(audio_path, sr=16000, mono=True)
            audio_array = np.asarray(audio_fallback, dtype=np.float32)
        except Exception:
            return None

    audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
    if audio_array.size == 0:
        return None

    peak = float(np.max(np.abs(audio_array)))
    if peak > 0.0:
        audio_array = audio_array / peak

    return audio_array


class ExternalASRDataset(Dataset):
    """Maps rows of the external transcript dataframe to Whisper model inputs."""

    def __init__(self, dataframe: pd.DataFrame, audio_dir: str, processor: AutoProcessor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]
        file_name = str(row["file_name"])
        transcript = str(row["transcript"])
        audio_path = os.path.join(self.audio_dir, file_name)

        audio = load_audio_16k(audio_path)
        if audio is None:
            raise RuntimeError(f"Failed to decode audio file: {audio_path}")

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids[0]
        audio_duration = float(len(audio) / 16000.0)

        return {
            "input_features": input_features,
            "labels": labels,
            "audio_duration": audio_duration,
        }


class ExternalASRDataModule(pl.LightningDataModule):
    """Loads an external transcript spreadsheet and filters it to valid, decodable samples."""

    def __init__(self):
        super().__init__()
        self.audio_dir = Config.EXTERNAL_AUDIO_DIR
        self.transcript_path = Config.EXTERNAL_TRANSCRIPT_PATH
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        self.collator = WhisperDataCollator(self.processor)
        self.test_dataset: ExternalASRDataset | None = None
        self.total_rows = 0
        self.used_rows = 0
        self.skipped_missing_audio = 0
        self.skipped_empty_transcript = 0
        self.skipped_invalid_audio = 0

    def _resolve_columns(self, dataframe: pd.DataFrame) -> tuple[str, str]:
        """Find the audio filename and transcript columns regardless of naming."""
        columns_map = {str(col).strip().lower(): str(col) for col in dataframe.columns}

        file_candidates = ["file_name", "filename", "file", "audio", "audio_file", "audio_name"]
        text_candidates = ["transcript", "text", "label", "sentence"]

        file_col = next((columns_map[c] for c in file_candidates if c in columns_map), None)
        text_col = next((columns_map[c] for c in text_candidates if c in columns_map), None)

        if file_col is None or text_col is None:
            raise ValueError(
                f"Could not resolve transcript columns. Found columns: {list(dataframe.columns)}"
            )
        return file_col, text_col

    def setup(self, stage: str | None = None) -> None:
        """Read the transcript file and build the filtered test dataset."""
        dataframe = pd.read_excel(self.transcript_path)
        file_col, text_col = self._resolve_columns(dataframe)
        self.total_rows = int(len(dataframe))

        rows: list[dict[str, str]] = []
        skipped_missing_audio = 0
        skipped_empty_transcript = 0
        skipped_invalid_audio = 0

        for _, row in dataframe.iterrows():
            raw_file_name = row[file_col]
            raw_text = row[text_col]

            if pd.isna(raw_file_name):
                skipped_missing_audio += 1
                continue

            file_name = str(raw_file_name).strip()
            if not file_name:
                skipped_missing_audio += 1
                continue
            if os.path.splitext(file_name)[1] == "":
                file_name = f"{file_name}.wav"

            transcript = "" if pd.isna(raw_text) else normalize_text(str(raw_text))
            if not transcript:
                skipped_empty_transcript += 1
                continue

            audio_path = os.path.join(self.audio_dir, file_name)
            if not os.path.isfile(audio_path):
                skipped_missing_audio += 1
                continue

            audio = load_audio_16k(audio_path)
            if audio is None:
                skipped_invalid_audio += 1
                continue

            rows.append({"file_name": file_name, "transcript": transcript})

        filtered_df = pd.DataFrame(rows)
        if filtered_df.empty:
            raise ValueError("No valid external test samples found after filtering.")

        self.test_dataset = ExternalASRDataset(filtered_df, self.audio_dir, self.processor)
        self.used_rows = int(len(filtered_df))
        self.skipped_missing_audio = skipped_missing_audio
        self.skipped_empty_transcript = skipped_empty_transcript
        self.skipped_invalid_audio = skipped_invalid_audio

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule is not set up. Call setup() before requesting test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collator,
        )


def run_external_test() -> None:
    """Evaluate Config.CKPT_PATH on the external test set and report metrics."""
    seed_all(Config.SEED)

    datamodule = ExternalASRDataModule()
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
        output_path = os.path.join(Config.OUTPUT_DIR, "external_test_metrics.json")

        payload = {
            "wer": float(wer),
            "cer": float(cer),
            "rtf": float(rtf),
            "total_rows": datamodule.total_rows,
            "used_rows": datamodule.used_rows,
            "skipped_missing_audio": datamodule.skipped_missing_audio,
            "skipped_empty_transcript": datamodule.skipped_empty_transcript,
            "skipped_invalid_audio": datamodule.skipped_invalid_audio,
            "audio_dir": datamodule.audio_dir,
            "transcript_path": datamodule.transcript_path,
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        print(f"External Test WER: {wer:.6f}")
        print(f"External Test CER: {cer:.6f}")
        print(f"External Test RTF: {rtf:.6f}")
        print(f"Total rows: {datamodule.total_rows}")
        print(f"Used rows: {datamodule.used_rows}")
        print(f"Skipped missing audio: {datamodule.skipped_missing_audio}")
        print(f"Skipped empty transcript: {datamodule.skipped_empty_transcript}")
        print(f"Skipped invalid audio: {datamodule.skipped_invalid_audio}")
        print(f"Saved metrics JSON: {output_path}")


if __name__ == "__main__":
    run_external_test()