from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from collections import deque

import numpy as np
import sounddevice as sd
import torch

from .dsp import STFTParams, TorchSTFT, apply_complex_mask, apply_deep_filter, hybrid_dsp_enhance, log_mag_features
from .model import AdaptiveDereverbNet, ModelConfig
from .utils import load_config, resolve_device


class RealtimeEnhancer:
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = resolve_device(config)
        self.sr = config["project"]["sample_rate"]
        self.hop = config["stft"]["hop_length"]
        self.context_seconds = config["realtime"]["context_seconds"]
        self.context_len = int(self.context_seconds * self.sr)
        self.context_len = max(self.context_len, 4 * self.hop)
        self.context = deque([0.0] * self.context_len, maxlen=self.context_len)
        self.input_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self.output_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.u_value = float(config["realtime"]["default_u"])
        self.use_model = bool(config["realtime"]["use_model"])
        self.processing_ms = 0.0

        self.stft = TorchSTFT(
            STFTParams(
                sample_rate=self.sr,
                n_fft=config["stft"]["n_fft"],
                win_length=config["stft"]["win_length"],
                hop_length=config["stft"]["hop_length"],
                window=config["stft"]["window"],
            ),
            device=self.device,
        )
        self.model = AdaptiveDereverbNet(ModelConfig(**config["model"])).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def set_u(self, value: float) -> None:
        self.u_value = float(value)

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        mono = indata[:, 0].copy()
        try:
            self.input_q.put_nowait(mono)
        except queue.Full:
            pass
        try:
            out = self.output_q.get_nowait()
        except queue.Empty:
            out = np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = out[:frames]

    def process_loop(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            for sample in chunk:
                self.context.append(float(sample))
            context_np = np.array(self.context, dtype=np.float32)
            u = torch.tensor([self.u_value], device=self.device)

            start = time.perf_counter()
            with torch.no_grad():
                x = torch.from_numpy(context_np).unsqueeze(0).to(self.device)
                Y = self.stft.stft(x)
                Y_dsp = hybrid_dsp_enhance(Y, u)
                X_hat = Y_dsp
                if self.use_model:
                    feats = log_mag_features(Y_dsp)
                    out = self.model(feats, u)
                    X_hat = apply_complex_mask(Y_dsp, out["mask"])
                    if self.config["model"]["use_deep_filter"]:
                        X_hat = X_hat + apply_deep_filter(Y_dsp, out["filter"])
                enhanced = self.stft.istft(X_hat, length=context_np.shape[0]).squeeze(0).cpu().numpy()
                out_chunk = enhanced[-len(chunk) :].astype(np.float32)
            self.processing_ms = (time.perf_counter() - start) * 1000.0

            try:
                self.output_q.put_nowait(out_chunk)
            except queue.Full:
                pass

    def run(self):
        worker = threading.Thread(target=self.process_loop, daemon=True)
        worker.start()

        root = tk.Tk()
        root.title("Adaptive Dereverberation Control")
        root.geometry("420x220")

        title = tk.Label(root, text="User-Controllable Dereverberation", font=("Arial", 14, "bold"))
        title.pack(pady=10)

        value_label = tk.Label(root, text=f"u = {self.u_value:.2f}", font=("Arial", 12))
        value_label.pack(pady=5)

        latency_label = tk.Label(root, text="Processing: 0.0 ms", font=("Arial", 11))
        latency_label.pack(pady=5)

        def on_slide(v):
            self.set_u(float(v))
            value_label.config(text=f"u = {float(v):.2f}")

        slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=320, command=on_slide)
        slider.set(self.u_value)
        slider.pack(pady=10)

        mode_var = tk.BooleanVar(value=self.use_model)

        def on_toggle():
            self.use_model = bool(mode_var.get())

        chk = tk.Checkbutton(root, text="Enable neural enhancement", variable=mode_var, command=on_toggle)
        chk.pack(pady=5)

        def refresh_labels():
            latency_label.config(text=f"Processing: {self.processing_ms:.1f} ms")
            if not self.stop_event.is_set():
                root.after(100, refresh_labels)

        root.after(100, refresh_labels)

        stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.hop,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        )

        def on_close():
            self.stop_event.set()
            stream.close()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        stream.start()
        root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    app = RealtimeEnhancer(config, args.checkpoint)
    app.run()


if __name__ == "__main__":
    main()
