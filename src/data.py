from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from .utils import (
    crop_or_pad,
    list_audio_files,
    normalize_peak,
    scale_noise_to_snr,
    try_read_audio,
)


@dataclass
class DatasetConfig:
    sample_rate: int
    segment_seconds: float
    clean_dir: str
    noise_dir: str
    rir_dir: str
    snr_db_range: Tuple[float, float]
    t60_range: Tuple[float, float]
    u_range: Tuple[float, float]


class DNSOnTheFlyDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, length: int = 4000):
        self.cfg = cfg
        self.length = length
        self.segment_len = int(cfg.sample_rate * cfg.segment_seconds)

        self.clean_files = list_audio_files(cfg.clean_dir)
        self.noise_files = list_audio_files(cfg.noise_dir)

        rir_candidates = list_audio_files(cfg.rir_dir)
        self.rir_files = self._filter_valid_rirs(rir_candidates)

        if not self.clean_files:
            raise RuntimeError(f"No clean files found in: {cfg.clean_dir}")
        if not self.noise_files:
            raise RuntimeError(f"No noise files found in: {cfg.noise_dir}")
        if not self.rir_files:
            raise RuntimeError(f"No readable RIR files found in: {cfg.rir_dir}")

        print(f"[Dataset] clean files: {len(self.clean_files)}")
        print(f"[Dataset] noise files: {len(self.noise_files)}")
        print(f"[Dataset] readable RIR files: {len(self.rir_files)} / {len(rir_candidates)}")

    def __len__(self) -> int:
        return self.length

    def _filter_valid_rirs(self, rir_candidates):
        valid = []
        for path in rir_candidates:
            rir = try_read_audio(path, self.cfg.sample_rate)
            if rir is None:
                continue
            if len(rir) < 16:
                continue
            if not np.all(np.isfinite(rir)):
                continue
            if np.max(np.abs(rir)) < 1e-8:
                continue
            valid.append(path)
        return valid

    def _sample_control_u(self) -> float:
        lo, hi = self.cfg.u_range
        return random.uniform(lo, hi)

    def _shape_rir(self, rir: np.ndarray, u: float, early_ms: float = 50.0) -> np.ndarray:
        sr = self.cfg.sample_rate
        early_n = int(sr * early_ms / 1000.0)
        rir = rir.astype(np.float32, copy=True)

        if len(rir) <= early_n + 1:
            return rir

        peak_idx = int(np.argmax(np.abs(rir)))
        tail_start = min(len(rir) - 1, peak_idx + early_n)
        tail_len = len(rir) - tail_start

        if tail_len <= 1:
            return rir

        t = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)

        # Larger u => stronger late-tail suppression
        alpha = 0.4 + 5.0 * float(u)
        tail_gain = np.exp(-alpha * t).astype(np.float32)

        rir[tail_start:] *= tail_gain
        return rir

    def _load_valid_triplet(self):
        for _ in range(20):
            clean_path = random.choice(self.clean_files)
            noise_path = random.choice(self.noise_files)
            rir_path = random.choice(self.rir_files)

            clean = try_read_audio(clean_path, self.cfg.sample_rate)
            noise = try_read_audio(noise_path, self.cfg.sample_rate)
            rir = try_read_audio(rir_path, self.cfg.sample_rate)

            if clean is None or len(clean) < 16:
                continue
            if noise is None or len(noise) < 16:
                continue
            if rir is None or len(rir) < 16:
                continue

            if not np.all(np.isfinite(clean)):
                continue
            if not np.all(np.isfinite(noise)):
                continue
            if not np.all(np.isfinite(rir)):
                continue

            if np.max(np.abs(rir)) < 1e-8:
                continue

            return clean, noise, rir

        raise RuntimeError("Failed to load a valid clean/noise/RIR triplet after 20 retries.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        clean, noise, rir = self._load_valid_triplet()

        clean = crop_or_pad(clean, self.segment_len)
        noise = crop_or_pad(noise, self.segment_len)

        u = self._sample_control_u()
        snr_db = random.uniform(*self.cfg.snr_db_range)

        rir = rir.astype(np.float32)
        rir = rir / (np.linalg.norm(rir) + 1e-8)

        shaped_rir = self._shape_rir(rir, u)

        reverberant = fftconvolve(clean, rir, mode="full")[: len(clean)].astype(np.float32)
        target = fftconvolve(clean, shaped_rir, mode="full")[: len(clean)].astype(np.float32)

        noise = scale_noise_to_snr(reverberant, noise, snr_db).astype(np.float32)
        noisy_reverb = reverberant + noise

        noisy_reverb = normalize_peak(noisy_reverb)
        target = normalize_peak(target)
        clean = normalize_peak(clean)

        return {
            "mixture": torch.from_numpy(noisy_reverb.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "clean": torch.from_numpy(clean.astype(np.float32)),
            "u": torch.tensor(u, dtype=torch.float32),
        }