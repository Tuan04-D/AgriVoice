"""Dataset, collator, and Lightning DataModule for the Hmong ASR train/val/test split."""

import os
from typing import Any

import librosa
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from config import Config


class HmongASRDataset(Dataset):
    """Maps rows of a labeled dataframe to Whisper input features and label ids."""

    def __init__(self, dataframe: pd.DataFrame, data_dir: str, processor: AutoProcessor):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]
        rel_audio_path = str(row["file_path"])
        label = str(row["label"])
        audio_path = os.path.join(self.data_dir, rel_audio_path)

        audio, _ = librosa.load(audio_path, sr=16000)
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(label, return_tensors="pt").input_ids[0]
        audio_duration = float(len(audio) / 16000.0)

        return {
            "input_features": input_features,
            "labels": labels,
            "audio_duration": audio_duration,
        }


class WhisperDataCollator:
    """Pads a batch of variable-length input features and label sequences."""

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        audio_duration = torch.tensor(
            [f["audio_duration"] for f in features], dtype=torch.float32
        )

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            labels,
            return_tensors="pt",
        )
        labels_ids = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        batch["labels"] = labels_ids
        batch["audio_duration"] = audio_duration
        return batch


class HmongASRDataModule(pl.LightningDataModule):
    """Loads label.csv and builds an 80/10/10 train/val/test split."""

    def __init__(self):
        super().__init__()
        self.data_dir = Config.DATA_DIR
        self.label_path = os.path.join(self.data_dir, Config.LABEL_FILE)
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        self.collator = WhisperDataCollator(self.processor)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Read the label file and split it into train, validation, and test sets."""
        dataframe = pd.read_csv(self.label_path)

        train_df, temp_df = train_test_split(
            dataframe,
            test_size=0.2,
            random_state=Config.SEED,
            shuffle=True,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=Config.SEED,
            shuffle=True,
        )

        self.train_dataset = HmongASRDataset(train_df, self.data_dir, self.processor)
        self.val_dataset = HmongASRDataset(val_df, self.data_dir, self.processor)
        self.test_dataset = HmongASRDataset(test_df, self.data_dir, self.processor)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collator,
        )
