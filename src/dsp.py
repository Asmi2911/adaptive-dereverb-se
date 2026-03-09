from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as Fnn


@dataclass
class STFTParams:
    sample_rate: int = 48000
    n_fft: int = 1024
    win_length: int = 960
    hop_length: int = 480
    window: str = "hann"


class TorchSTFT:
    def __init__(self, params: STFTParams, device: torch.device | str = "cpu"):
        self.params = params
        self.device = torch.device(device)
        if params.window != "hann":
            raise ValueError("Only hann window is currently supported.")
        self.window = torch.hann_window(params.win_length, device=self.device)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.stft(
            x,
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
            win_length=self.params.win_length,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )

    def istft(self, X: torch.Tensor, length: int | None = None) -> torch.Tensor:
        return torch.istft(
            X,
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
            win_length=self.params.win_length,
            window=self.window,
            center=True,
            length=length,
        )


def mag_phase(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mag = torch.abs(X)
    phase = torch.angle(X)
    return mag, phase


def complex_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return torch.polar(mag, phase)


def log_mag_features(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mag = torch.abs(X)
    return torch.log(mag + eps)


def apply_complex_mask(Y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return Y * mask


def apply_deep_filter(Y: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """
    Y: [B, F, T] complex
    filt: [B, F, T, K] complex
    Returns filtered spectrum [B, F, T]
    """
    B, Fbins, T = Y.shape
    K = filt.shape[-1]
    out = torch.zeros_like(Y)
    for k in range(K):
        if k == 0:
            delayed = Y
        else:
            delayed = Fnn.pad(Y[..., :-k], (k, 0))
        out = out + filt[..., k] * delayed
    return out


def spectral_wiener_gain(Y_mag: torch.Tensor, alpha: float = 0.98, floor: float = 0.05) -> torch.Tensor:
    noise_psd = torch.zeros_like(Y_mag[:, :, :1])
    gains = []
    for t in range(Y_mag.shape[-1]):
        cur = Y_mag[:, :, t : t + 1] ** 2
        noise_psd = alpha * noise_psd + (1.0 - alpha) * cur
        post_snr = cur / (noise_psd + 1e-8)
        gain = post_snr / (1.0 + post_snr)
        gain = torch.clamp(gain, min=floor, max=1.0)
        gains.append(gain)
    return torch.cat(gains, dim=-1)


def late_reverb_suppression(Y: torch.Tensor, u: torch.Tensor, decay_power: float = 1.4) -> torch.Tensor:
    """
    Simple interpretable DSP baseline.
    Suppresses temporally diffuse late energy using a frame-wise
    backward energy ratio. Higher u => stronger suppression.
    u shape: [B] or scalar tensor
    """
    mag2 = torch.abs(Y) ** 2
    tail = torch.flip(torch.cumsum(torch.flip(mag2, dims=[-1]), dim=-1), dims=[-1])
    local = Fnn.avg_pool1d(mag2, kernel_size=5, stride=1, padding=2)
    ratio = local / (tail + 1e-6)
    ratio = torch.clamp(ratio, 0.0, 1.0)
    if u.dim() == 0:
        u = u.view(1)
    u = u.view(-1, 1, 1)
    gain = torch.clamp(1.0 - u * (1.0 - ratio) ** decay_power, 0.15, 1.0)
    return Y * gain


def hybrid_dsp_enhance(Y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    Y_mag = torch.abs(Y)
    gain = spectral_wiener_gain(Y_mag)
    X = Y * gain
    X = late_reverb_suppression(X, u=u)
    return X
