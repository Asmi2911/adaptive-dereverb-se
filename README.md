# Adaptive Dereverberation Speech Enhancement

This codebase implements a project-aligned prototype for **user-controllable, low-latency speech enhancement with adaptive dereverberation using a hybrid DSP-neural pipeline**.

## What this repo contains

- **Streaming STFT / iSTFT DSP engine**
- **DSP baselines** for noise suppression and late-reverb suppression
- **Lightweight GRU-based neural enhancer**
- **Optional deep filtering head** for multi-tap complex spectral filtering
- **Continuous dereverberation control `u ∈ [0, 1]`**
- **DNS-style on-the-fly data synthesis** from clean speech + noise + RIRs
- **Offline inference** on `.wav` files
- **Real-time microphone demo** with a Tkinter slider
- **Evaluation script** with SI-SDR and optional PESQ/STOI

## Expected folder structure

You can keep your current layout, but the cleanest structure is:

```text
adaptive_dereverb_se/
├── config.yaml
├── requirements.txt
├── noise_fullband/
├── datasets_fullband/
│   ├── clean_fullband/
│   │   └── vctk_wav48_silence_trimmed/
│   └── impulse_responses/
│       └── SLR26/simulated_rirs_48k/smallroom/
└── src/
```

If you currently have both:

- `datasets_fullband/noise_fullband/`
- `noise_fullband/`

keep **one canonical copy** only. The config currently points to the root-level `noise_fullband/`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config config.yaml
```

## Offline inference

```bash
python -m src.infer_file --config config.yaml --input input.wav --output outputs/enhanced.wav --checkpoint checkpoints/best.pt --u 0.6
```

## Real-time demo

```bash
python -m src.realtime_app --config config.yaml --checkpoint checkpoints/best.pt
```

## Evaluate

```bash
python -m src.eval_metrics --clean clean.wav --estimate outputs/enhanced.wav
```

## Notes

- This is a strong research-grade prototype, but real-time stability still depends on your laptop CPU and audio driver settings.
- Start with **offline inference first**, then try the real-time GUI.
- If real-time is glitchy, reduce model size or disable `use_deep_filter` in `config.yaml`.
<img width="1647" height="550" alt="image" src="https://github.com/user-attachments/assets/a7dacefe-3542-44d2-9020-d9dc38a48ee6" />

