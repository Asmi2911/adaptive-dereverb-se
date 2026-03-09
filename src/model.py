from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    freq_bins: int = 513
    hidden_size: int = 160
    num_layers: int = 2
    dropout: float = 0.1
    deep_filter_taps: int = 3
    use_deep_filter: bool = True


class AdaptiveDereverbNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        input_size = cfg.freq_bins + 1  # log-mag features + scalar u
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.PReLU(),
        )
        self.rnn = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.post = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.PReLU(),
        )
        self.mask_head = nn.Linear(cfg.hidden_size, cfg.freq_bins * 2)
        self.filter_head = nn.Linear(cfg.hidden_size, cfg.freq_bins * cfg.deep_filter_taps * 2)

    def forward(self, log_mag: torch.Tensor, u: torch.Tensor) -> dict:
        """
        log_mag: [B, F, T]
        u: [B]
        """
        B, F, T = log_mag.shape
        x = log_mag.transpose(1, 2)  # [B, T, F]
        u_feat = u.view(B, 1, 1).expand(B, T, 1)
        x = torch.cat([x, u_feat], dim=-1)
        x = self.input_proj(x)
        x, _ = self.rnn(x)
        x = self.post(x)

        mask = self.mask_head(x).view(B, T, F, 2)
        mask = torch.tanh(mask)
        mask = torch.complex(mask[..., 0], mask[..., 1]).permute(0, 2, 1)

        filt = self.filter_head(x).view(B, T, F, self.cfg.deep_filter_taps, 2)
        filt = torch.tanh(filt)
        filt = torch.complex(filt[..., 0], filt[..., 1]).permute(0, 2, 1, 3)

        return {"mask": mask, "filter": filt}
