"""Central configuration for training, evaluation, and inference."""


class Config:
    """Hyperparameters and filesystem paths shared across all scripts.

    Path fields are placeholders and must be updated to match the
    target machine before running any script.
    """

    # Directory containing the preprocessed dataset: audio segments plus LABEL_FILE.
    DATA_DIR = "/path/to/dataset-preprocessed"
    LABEL_FILE = "label.csv"

    # Directory where checkpoints, TensorBoard logs, and metrics are written.
    OUTPUT_DIR = "/path/to/outputs"

    MODEL_NAME = "openai/whisper-small"
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    LR = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500

    MAX_EPOCHS = 500
    PATIENCE = 10
    PRECISION = "bf16-mixed"

    SEED = 42
    RUN_TEST_AFTER_TRAIN = True

    # Checkpoint used for resuming training, and for test/inference scripts.
    # Point this at a .ckpt file produced by train.py under OUTPUT_DIR.
    CKPT_PATH = "/path/to/outputs/whisper_small_ft/version_0/checkpoints/best.ckpt"

    # Real-world (out-of-domain) test set used to measure generalization,
    # separate from the internal train/val/test split drawn from DATA_DIR.
    EXTERNAL_AUDIO_DIR = "/path/to/external-test/audio"
    EXTERNAL_TRANSCRIPT_PATH = "/path/to/external-test/transcripts.xlsx"
