# Finetune Whisper ASR — Hmong

Fine-tunes `openai/whisper-small` for Automatic Speech Recognition (ASR) in
Hmong, part of the HmongBridge project.

## Data source

The dataset was built from publicly available Hmong-language religious audio
recordings (the Hmong Bible, Old and New Testaments) with corresponding
per-chapter transcripts. The raw data consists of long-form audio (MP3,
converted to 16 kHz mono WAV with `ffmpeg`) paired with TXT transcripts.

This repository only covers the **fine-tuning** stage. It assumes the data
has already been preprocessed into a `label.csv` file plus a directory of
segmented audio clips. The crawling and preprocessing scripts (forced
alignment, segmentation) are not part of this repository.

## Preprocessing (summary)

Long-form audio is segmented into short clips using word-level forced
alignment (Whisper large via `stable-whisper`), avoiding the mid-sentence or
mid-word cuts that fixed-length segmentation would produce. A new segment is
finalized once the accumulated duration reaches 25 seconds or the current
word ends with punctuation. Each audio segment is saved as its own WAV file,
with the corresponding text label written to `label.csv` under two columns,
`file_path` and `label`.

## Dataset statistics

| Property | Value |
|---|---|
| Total utterances | 36,065 |
| Total duration | 48.70 hours |
| Average duration / utterance | 4.86 seconds |
| Total words | 445,124 |
| Unique words | 8,788 |
| Average words / utterance | 12.34 |

The data is randomly split 80/10/10 (seed = 42):

| Split | Utterances | Duration (hours) |
|---|---|---|
| Train | 28,852 | ≈38.96 |
| Validation | 3,607 | ≈4.87 |
| Internal test | 3,606 | ≈4.87 |

All data comes from a single domain (read religious audio), so the internal
test set shares the same acoustic distribution as the training set — an
independent, real-world test set is needed to assess generalization (see
Evaluation below).

## Fine-tuning pipeline

- **Base model:** `openai/whisper-small` (~244M parameters), full
  fine-tuning of all parameters via `WhisperForConditionalGeneration`
  (Hugging Face Transformers).
- **Framework:** PyTorch Lightning, tracked with TensorBoard.
- **Optimizer:** AdamW, lr = 1e-5, weight decay = 0.01.
- **LR schedule:** linear warmup (500 steps) + linear decay.
- **Batch size:** 16/device, gradient accumulation over 4 steps → effective
  batch size 64.
- **Precision:** bf16-mixed.
- **Max epochs:** 500, early stopping with patience 10 epochs on `val_wer`.
- **Checkpointing:** keeps the checkpoint with the best `val_wer`.
- **Seed:** 42.

The best model was obtained at epoch 21 (val_wer = 12.26%).

## Evaluation results

**Internal test** (same distribution as training data):

| Metric | Value |
|---|---|
| WER | 12.35% |
| CER | 8.68% |
| RTF | 0.0232 |

**Real-world test** (self-recorded audio, uncontrolled conditions):

| Metric | Value |
|---|---|
| WER | 26.60% |
| CER | 12.02% |
| RTF | 0.0125 |

RTF is very low on both sets (< 0.03), meaning the model processes audio
40–80x faster than real time, which comfortably meets real-world deployment
speed requirements.

The WER gap between the two sets (+14.25 points) reflects a clear domain
shift: the training data comes from a narrow domain (slow, clear, low-noise
religious audio), while real-world audio has background noise, different
recording device characteristics, and more natural speaking pace/style. The
WER gap is much larger than the CER gap (+3.34 points), which suggests the
model still recognizes basic acoustic units well but struggles to assemble
them into correct words — typical of out-of-vocabulary errors combined with
word-boundary errors.

Full details (methodology, formulas, analysis) are in
`technical_report.tex` at the root of the `finetune-whisper/` directory.

## Directory structure

```
finetune-whisper-asr-hmong/
├── scripts/
│   ├── config.py                  # data/output paths and hyperparameters
│   ├── data.py                    # Dataset/DataModule, feature extraction, 80/10/10 split
│   ├── train.py                   # LightningModule + training loop
│   ├── test.py                    # evaluation on the internal test set
│   ├── external_test.py           # evaluation on the real-world test set
│   ├── manual_check_external.py   # generates a predict/true comparison CSV, flags hallucinations
│   ├── test_inference.py          # ad hoc inference via microphone
│   └── utils.py                   # seeding, text normalization (normalize_text)
├── outputs/
│   ├── dataset_stats.txt              # overall dataset statistics
│   ├── test_metrics.json              # WER/CER/RTF — internal test
│   ├── external_test_metrics.json     # WER/CER/RTF — real-world test
│   ├── external_manual_check.csv      # detailed per-sample predict/true comparison
│   ├── val_cer.png / val_wer.png      # CER/WER curves over training steps
│   └── whisper_small_ft/version_0/
│       ├── hparams.yaml               # logged hyperparameters
│       └── events.out.tfevents...     # TensorBoard log
└── requirements.txt
```

The model checkpoint (`.ckpt`) is not included in the repository due to its
size.

## Running

Before running any script, edit `scripts/config.py` and replace the
placeholder paths (`DATA_DIR`, `OUTPUT_DIR`, `CKPT_PATH`,
`EXTERNAL_AUDIO_DIR`, `EXTERNAL_TRANSCRIPT_PATH`) with paths valid on your
machine.

```bash
cd scripts
python train.py                    # fine-tune, then auto-run the internal test
python test.py                      # evaluate on the internal test set only
python external_test.py             # evaluate on the real-world test set
python manual_check_external.py     # generate a detailed comparison report
```

The expected input is a directory of segmented audio clips together with
`label.csv` (columns `file_path`, `label`).
