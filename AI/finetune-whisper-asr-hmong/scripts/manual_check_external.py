"""Generate a per-sample prediction report to spot hallucinations on the external test set."""

import os
import unicodedata

import jiwer
import pandas as pd
import torch

from config import Config
from external_test import ExternalASRDataModule, load_audio_16k
from train import WhisperLightningModule
from utils import normalize_text, seed_all


def strip_accents(text: str) -> str:
    """Remove combining diacritical marks from text."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def repetition_stats(text: str) -> tuple[bool, int, str]:
    """Detect runs of 3+ consecutive repeated tokens, a common decoding hallucination."""
    tokens = text.split()
    if not tokens:
        return False, 0, ""

    max_run = 1
    current_run = 1
    current_token = tokens[0]
    run_token = tokens[0]

    for token in tokens[1:]:
        if token == current_token:
            current_run += 1
        else:
            current_token = token
            current_run = 1

        if current_run > max_run:
            max_run = current_run
            run_token = current_token

    return max_run >= 3, max_run, run_token


def has_unwanted_insertions(true_text: str, pred_text: str) -> bool:
    """Flag predictions that insert religious-domain keywords absent from the reference."""
    true_ascii = strip_accents(true_text.lower())
    pred_ascii = strip_accents(pred_text.lower())
    keywords = ["vajtswv", "chua"]
    for kw in keywords:
        if kw in pred_ascii and kw not in true_ascii:
            return True
    return False


def run_manual_check_external() -> None:
    """Run inference on the external set and write a per-sample comparison CSV."""
    seed_all(Config.SEED)

    datamodule = ExternalASRDataModule()
    datamodule.setup()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lightning_module = WhisperLightningModule.load_from_checkpoint(Config.CKPT_PATH)
    model = lightning_module.model.to(device)
    model.eval()
    processor = datamodule.processor

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for _, row in datamodule.test_dataset.dataframe.iterrows():
            file_name = str(row["file_name"])
            true_text = str(row["transcript"])
            audio_path = os.path.join(datamodule.audio_dir, file_name)
            audio = load_audio_16k(audio_path)
            if audio is None:
                continue

            input_features = processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device)

            generated_ids = model.generate(input_features=input_features)
            pred_raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred_text = normalize_text(pred_raw)

            sample_wer = float(jiwer.wer([true_text], [pred_text]))
            sample_cer = float(jiwer.cer([true_text], [pred_text]))
            is_repeat, max_run, repeat_token = repetition_stats(pred_text)
            unwanted_insert = has_unwanted_insertions(true_text, pred_text)

            rows.append(
                {
                    "file_name": file_name,
                    "true_label": true_text,
                    "predict": pred_text,
                    "wer": sample_wer,
                    "cer": sample_cer,
                    "is_repetition_hallucination": is_repeat,
                    "max_repetition_run": max_run,
                    "repetition_token": repeat_token,
                    "has_unwanted_vajtswv_or_chua": unwanted_insert,
                }
            )

    if not rows:
        raise RuntimeError("No valid rows to evaluate in external dataset.")

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["wer", "cer"], ascending=False).reset_index(drop=True)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(Config.OUTPUT_DIR, "external_manual_check.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    top_k = min(5, len(df))
    repeated_count = int(df["is_repetition_hallucination"].sum())
    unwanted_count = int(df["has_unwanted_vajtswv_or_chua"].sum())

    print(f"Saved manual check CSV: {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"Average WER: {df['wer'].mean():.6f}")
    print(f"Average CER: {df['cer'].mean():.6f}")
    print(f"Repetition hallucination count: {repeated_count}")
    print(f"Unwanted Vajtswv/Chua insertion count: {unwanted_count}")

    print(f"Top {top_k} worst samples:")
    for i in range(top_k):
        rec = df.iloc[i]
        print(f"[{i + 1}] file: {rec['file_name']}")
        print(f"WER={rec['wer']:.6f}, CER={rec['cer']:.6f}")
        print(f"TRUE: {rec['true_label']}")
        print(f"PRED: {rec['predict']}")

    if repeated_count > 0:
        print("Model shows repetition hallucination on some samples.")
    else:
        print("No repetition hallucination detected by the consecutive-token rule.")

    if unwanted_count > 0:
        print("Model inserted Vajtswv/Chua in places where reference did not contain them.")
    else:
        print("No unwanted Vajtswv/Chua insertion detected.")


if __name__ == "__main__":
    run_manual_check_external()