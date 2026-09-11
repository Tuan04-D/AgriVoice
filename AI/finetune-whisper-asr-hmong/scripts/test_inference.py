"""Record 5 seconds of microphone audio and transcribe it with a fine-tuned checkpoint."""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoProcessor

from config import Config
from train import WhisperLightningModule
from utils import seed_all

try:
    import sounddevice as sd
except ImportError as import_error:
    print(
        "Missing dependency: sounddevice. Install with 'pip install sounddevice'.",
        file=sys.stderr,
    )
    raise SystemExit(1) from import_error


SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def _list_input_devices() -> list[tuple[int, str, int]]:
    """List available audio devices that support input (index, name, channel count)."""
    devices = sd.query_devices()
    input_devices: list[tuple[int, str, int]] = []
    for index, info in enumerate(devices):
        input_channels = int(info.get("max_input_channels", 0))
        if input_channels > 0:
            input_devices.append((index, str(info.get("name", f"device_{index}")), input_channels))
    return input_devices


def _resolve_input_device(preferred_device: int | None) -> int:
    """Validate the preferred device, or fall back to the default/first input device."""
    if preferred_device is not None:
        info = sd.query_devices(preferred_device)
        if int(info.get("max_input_channels", 0)) < 1:
            raise RuntimeError(
                f"Device {preferred_device} is not an input device. "
                "Use --list-devices to see valid microphone devices."
            )
        return preferred_device

    default_device = sd.default.device[0]
    if isinstance(default_device, (int, np.integer)) and int(default_device) >= 0:
        info = sd.query_devices(int(default_device))
        if int(info.get("max_input_channels", 0)) > 0:
            return int(default_device)

    input_devices = _list_input_devices()
    if input_devices:
        return input_devices[0][0]

    raise RuntimeError(
        "No microphone input device found. "
        "Please attach/map an input audio device to this environment, then try again."
    )


def _record_audio(input_device: int) -> np.ndarray:
    """Record RECORD_SECONDS of mono audio at SAMPLE_RATE from the given device."""
    frame_count = int(SAMPLE_RATE * RECORD_SECONDS)
    print(
        f"Recording from microphone (device={input_device}) for {RECORD_SECONDS} seconds..."
    )
    try:
        audio = sd.rec(
            frame_count,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=input_device,
        )
    except sd.PortAudioError as error:
        raise RuntimeError(
            f"Failed to open input device {input_device}: {error}. "
            "Use --list-devices and pick a valid input device with --device."
        ) from error
    sd.wait()
    print("Recording finished.")
    return np.squeeze(audio)


def run_inference(preferred_device: int | None = None) -> None:
    """Record from the microphone and print the model's transcription."""
    seed_all(Config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
    lightning_module = WhisperLightningModule.load_from_checkpoint(Config.CKPT_PATH)
    model = lightning_module.model.to(device)
    model.eval()

    input_device = _resolve_input_device(preferred_device)
    audio = _record_audio(input_device)
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(input_features=input_features)

    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("Predicted text:")
    print(prediction if prediction else "(empty output)")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for device selection and listing."""
    parser = argparse.ArgumentParser(
        description="Record 5 seconds from microphone and run Whisper inference."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input audio device index. If omitted, auto-select a valid microphone device.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list_devices:
        devices = _list_input_devices()
        if not devices:
            print("No input audio devices found.")
        else:
            print("Available input devices:")
            for index, name, channels in devices:
                print(f"- {index}: {name} (max_input_channels={channels})")
        raise SystemExit(0)

    try:
        run_inference(preferred_device=args.device)
    except RuntimeError as runtime_error:
        print(str(runtime_error), file=sys.stderr)
        raise SystemExit(1)