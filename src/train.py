from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DNSOnTheFlyDataset, DatasetConfig
from .dsp import STFTParams, TorchSTFT, apply_complex_mask, apply_deep_filter, hybrid_dsp_enhance, log_mag_features
from .model import AdaptiveDereverbNet, ModelConfig
from .utils import ensure_dir, load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    return parser.parse_args()


def build_objects(config: dict):
    device = resolve_device(config)
    stft_params = STFTParams(
        sample_rate=config["project"]["sample_rate"],
        n_fft=config["stft"]["n_fft"],
        win_length=config["stft"]["win_length"],
        hop_length=config["stft"]["hop_length"],
        window=config["stft"]["window"],
    )
    stft = TorchSTFT(stft_params, device=device)
    model_cfg = ModelConfig(**config["model"])
    model = AdaptiveDereverbNet(model_cfg).to(device)
    data_cfg = DatasetConfig(
        sample_rate=config["project"]["sample_rate"],
        segment_seconds=config["project"]["segment_seconds"],
        clean_dir=config["paths"]["clean_dir"],
        noise_dir=config["paths"]["noise_dir"],
        rir_dir=config["paths"]["rir_dir"],
        snr_db_range=tuple(config["train"]["snr_db_range"]),
        t60_range=tuple(config["train"]["t60_range"]),
        u_range=tuple(config["train"]["u_range"]),
    )
    ds = DNSOnTheFlyDataset(data_cfg, length=max(1000, config["train"]["batch_size"] * 100))
    loader = DataLoader(
        ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        drop_last=True,
    )
    return device, stft, model, loader


def complex_loss(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_hat.real, x_true.real) + F.l1_loss(x_hat.imag, x_true.imag)


def train() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(42)
    device, stft, model, loader = build_objects(config)

    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    ensure_dir(ckpt_dir)
    best_loss = float("inf")

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        pbar = tqdm(total=args.steps_per_epoch, desc=f"Epoch {epoch}")
        loader_iter = iter(loader)

        for _ in range(args.steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            mix = batch["mixture"].to(device)
            target = batch["target"].to(device)
            u = batch["u"].to(device)

            Y = stft.stft(mix)
            X = stft.stft(target)

            Y_dsp = hybrid_dsp_enhance(Y, u)
            feats = log_mag_features(Y_dsp)
            out = model(feats, u)

            X_hat = apply_complex_mask(Y_dsp, out["mask"])
            if config["model"]["use_deep_filter"]:
                X_hat = X_hat + apply_deep_filter(Y_dsp, out["filter"])

            wav_hat = stft.istft(X_hat, length=target.shape[-1])
            wav_tgt = target

            loss_spec = complex_loss(X_hat, X)
            loss_time = F.l1_loss(wav_hat, wav_tgt)
            loss = loss_spec + 0.5 * loss_time

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()

            running += float(loss.item())
            pbar.update(1)
            pbar.set_postfix(loss=f"{running / pbar.n:.4f}")

        pbar.close()
        avg_loss = running / args.steps_per_epoch

        last_ckpt = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": config,
                "avg_loss": avg_loss,
            },
            last_ckpt,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "avg_loss": avg_loss,
                },
                ckpt_dir / "best.pt",
            )

        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}, best_loss={best_loss:.4f}")


if __name__ == "__main__":
    train()
