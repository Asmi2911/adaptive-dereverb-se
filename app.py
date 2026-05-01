import os
import time
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt

from src.dsp import STFTParams, TorchSTFT, apply_complex_mask, apply_deep_filter, hybrid_dsp_enhance, log_mag_features
from src.model import AdaptiveDereverbNet, ModelConfig
from src.utils import load_config, read_audio, write_audio


CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = "checkpoints/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config(CONFIG_PATH)
sr = config["project"]["sample_rate"]

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

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
else:
    model = None


def plot_spectrogram(wav, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.specgram(wav, Fs=sr, NFFT=1024, noverlap=512)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def enhance_audio(audio_path, u, mode):
    if audio_path is None:
        raise gr.Error("Please upload a WAV file.")

    wav = read_audio(audio_path, sr)
    x = torch.from_numpy(wav).float().to(device).unsqueeze(0)
    u_tensor = torch.tensor([float(u)], device=device)

    start = time.perf_counter()

    with torch.no_grad():
        Y = stft.stft(x)
        Y_dsp = hybrid_dsp_enhance(Y, u_tensor)

        if mode == "DSP only" or model is None:
            X_hat = Y_dsp
        else:
            feats = log_mag_features(Y_dsp)
            out = model(feats, u_tensor)
            X_hat = apply_complex_mask(Y_dsp, out["mask"])
            if config["model"]["use_deep_filter"]:
                X_hat = X_hat + apply_deep_filter(Y_dsp, out["filter"])

        enhanced = stft.istft(X_hat, length=x.shape[-1]).squeeze(0).cpu().numpy()

    elapsed = time.perf_counter() - start
    duration = len(wav) / sr
    rtf = elapsed / max(duration, 1e-8)

    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    write_audio(output_path, enhanced, sr)

    before_plot = plot_spectrogram(wav, "Input Reverberant/Noisy Speech")
    after_plot = plot_spectrogram(enhanced, f"Enhanced Speech | u={u:.2f} | {mode}")

    summary = f"""
### Result Summary

| Item | Value |
|---|---:|
| Mode | {mode} |
| Dereverberation control `u` | {u:.2f} |
| Audio duration | {duration:.2f} sec |
| Processing time | {elapsed:.3f} sec |
| Real-time factor | {rtf:.3f} |

Lower RTF is better. RTF < 1 means faster than real time.
"""

    return output_path, before_plot, after_plot, summary


demo = gr.Interface(
    fn=enhance_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload reverberant/noisy WAV"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Dereverberation Strength u"),
        gr.Radio(["DSP only", "DSP + Neural"], value="DSP + Neural", label="Enhancement Mode"),
    ],
    outputs=[
        gr.Audio(type="filepath", label="Enhanced Output"),
        gr.Image(label="Input Spectrogram"),
        gr.Image(label="Enhanced Spectrogram"),
        gr.Markdown(label="Results"),
    ],
    title="Adaptive Speech Dereverberation",
    description="Hybrid DSP–Neural speech enhancement with user-controllable dereverberation.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)