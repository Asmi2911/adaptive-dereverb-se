from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .dsp import STFTParams, TorchSTFT, apply_complex_mask, apply_deep_filter, hybrid_dsp_enhance, log_mag_features
from .model import AdaptiveDereverbNet, ModelConfig
from .utils import ensure_dir, load_config, read_audio, resolve_device, write_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--u", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config)
    sr = config["project"]["sample_rate"]
    wav = read_audio(args.input, sr)

    stft = TorchSTFT(
        STFTParams(
            sample_rate=sr,
            n_fft=config["stft"]["n_fft"],
            win_length=config["stft"]["win_length"],
            hop_length=config["stft"]["hop_length"],
            window=config["stft"]["window"],
        ),
        device=device,
    )

    model = AdaptiveDereverbNet(ModelConfig(**config["model"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x = torch.from_numpy(wav).float().to(device).unsqueeze(0)
    u = torch.tensor([float(args.u)], device=device)

    start = time.perf_counter()
    with torch.no_grad():
        Y = stft.stft(x)
        Y_dsp = hybrid_dsp_enhance(Y, u)
        feats = log_mag_features(Y_dsp)
        out = model(feats, u)
        X_hat = apply_complex_mask(Y_dsp, out["mask"])
        if config["model"]["use_deep_filter"]:
            X_hat = X_hat + apply_deep_filter(Y_dsp, out["filter"])
        enhanced = stft.istft(X_hat, length=x.shape[-1]).squeeze(0).cpu().numpy()
    elapsed = time.perf_counter() - start

    ensure_dir(Path(args.output).parent)
    write_audio(args.output, enhanced, sr)
    audio_seconds = len(wav) / sr
    rtf = elapsed / max(audio_seconds, 1e-6)
    print(f"Saved enhanced file to {args.output}")
    print(f"Audio duration: {audio_seconds:.2f}s")
    print(f"Processing time: {elapsed:.3f}s")
    print(f"Real-time factor (RTF): {rtf:.3f}")


if __name__ == "__main__":
    main()
