"""Fine-tune openai/whisper-small on the Hmong ASR dataset with PyTorch Lightning."""

import os
import time

import jiwer
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from transformers import WhisperForConditionalGeneration, get_linear_schedule_with_warmup

from config import Config
from data import HmongASRDataModule
from utils import normalize_text, seed_all


class WhisperLightningModule(pl.LightningModule):
    """LightningModule wrapping WhisperForConditionalGeneration for ASR fine-tuning."""

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        lr: float = Config.LR,
        weight_decay: float = Config.WEIGHT_DECAY,
        warmup_steps: int = Config.WARMUP_STEPS,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the cross-entropy loss for a batch of features and target labels."""
        outputs = self.model(input_features=input_features, labels=labels)
        return outputs.loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Log validation loss, WER, and CER for one batch."""
        outputs = self.model(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        val_loss = outputs.loss

        generated_ids = self.model.generate(input_features=batch["input_features"])
        labels = batch["labels"].clone()
        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.eos_token_id
        labels[labels == -100] = pad_token_id

        pred_texts = self.trainer.datamodule.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        ref_texts = self.trainer.datamodule.processor.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        pred_texts = [normalize_text(text) for text in pred_texts]
        ref_texts = [normalize_text(text) for text in ref_texts]

        val_wer = jiwer.wer(ref_texts, pred_texts)
        val_cer = jiwer.cer(ref_texts, pred_texts)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_wer", val_wer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cer", val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Log test loss, WER, CER, and real-time factor (RTF) for one batch."""
        outputs = self.model(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        test_loss = outputs.loss

        start_time = time.perf_counter()
        generated_ids = self.model.generate(input_features=batch["input_features"])
        elapsed = time.perf_counter() - start_time

        labels = batch["labels"].clone()
        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.model.config.eos_token_id
        labels[labels == -100] = pad_token_id

        pred_texts = self.trainer.datamodule.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        ref_texts = self.trainer.datamodule.processor.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        pred_texts = [normalize_text(text) for text in pred_texts]
        ref_texts = [normalize_text(text) for text in ref_texts]

        test_wer = jiwer.wer(ref_texts, pred_texts)
        test_cer = jiwer.cer(ref_texts, pred_texts)

        audio_duration = float(batch["audio_duration"].sum().item())
        test_rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_wer", test_wer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_cer", test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_rtf", test_rtf, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, object]:
        """Build the AdamW optimizer and linear warmup/decay scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def run_training() -> None:
    """Fine-tune the model and, if configured, run the internal test afterward."""
    seed_all(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    datamodule = HmongASRDataModule()
    model = WhisperLightningModule()

    logger = TensorBoardLogger(
        save_dir=Config.OUTPUT_DIR,
        name="whisper_small_ft",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_wer:.4f}",
    )
    early_stopping = EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=Config.PATIENCE,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        max_epochs=Config.MAX_EPOCHS,
        precision=Config.PRECISION,
        accumulate_grad_batches=Config.GRAD_ACCUM_STEPS,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=Config.CKPT_PATH)

    if Config.RUN_TEST_AFTER_TRAIN:
        best_path = checkpoint_callback.best_model_path
        ckpt_path = best_path if best_path else None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    run_training()
