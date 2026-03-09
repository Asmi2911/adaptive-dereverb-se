from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import yaml


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def load_config(path: str | os.PathLike) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict) -> torch.device:
    device_name = config["project"].get("device", "auto")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_audio_files(root: str | os.PathLike) -> List[str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    files = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise FileNotFoundError(f"No audio files found under: {root}")

    return sorted(files)


def _resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)

    from scipy.signal import resample_poly

    g = math.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    audio = resample_poly(audio, up, down)
    return audio.astype(np.float32)


def _postprocess_audio(audio: np.ndarray) -> Optional[np.ndarray]:
    if audio is None:
        return None

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 0:
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        return None

    if not np.all(np.isfinite(audio)):
        return None

    max_abs = np.max(np.abs(audio)) + 1e-8
    if max_abs > 1.0:
        audio = audio / max_abs

    return audio.astype(np.float32)


def try_read_audio(path: str | os.PathLike, target_sr: int) -> Optional[np.ndarray]:
    """
    Safely read audio. Returns mono float32 waveform on success, else None.
    This prevents corrupt / unsupported files from crashing training.
    """
    try:
        audio, sr = sf.read(path, always_2d=False)
        audio = _postprocess_audio(audio)
        if audio is None:
            return None

        audio = _resample_audio(audio, sr, target_sr)
        audio = _postprocess_audio(audio)
        return audio
    except Exception:
        return None


def read_audio(path: str | os.PathLike, target_sr: int) -> np.ndarray:
    """
    Strict audio read. Raises a clear error if unreadable.
    """
    audio = try_read_audio(path, target_sr)
    if audio is None:
        raise RuntimeError(f"Unreadable or invalid audio file: {path}")
    return audio


def write_audio(path: str | os.PathLike, audio: np.ndarray, sr: int) -> None:
    ensure_dir(Path(path).parent)
    sf.write(path, np.clip(audio, -1.0, 1.0), sr)


def crop_or_pad(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == length:
        return x
    if len(x) > length:
        start = random.randint(0, len(x) - length)
        return x[start : start + length]
    reps = int(np.ceil(length / max(len(x), 1)))
    out = np.tile(x, reps)[:length]
    return out.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0) + 1e-12)
    return noise * (target_noise_rms / (noise_rms + 1e-12))


def normalize_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = np.max(np.abs(x)) + 1e-12
    return (x / max_abs * peak).astype(np.float32)


def to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))