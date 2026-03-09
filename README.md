# Adaptive Neural Speech Dereverberation & Enhancement

Hybrid **DSP + Neural Network pipeline** for adaptive speech dereverberation and noise suppression.

This project is part of an **Advanced Project at NYU Tandon School of Engineering**.

**Advisor:** Prof. Ivan Selesnick  
**Author:** Asmita Sonavane  

---

# Project Overview

Speech recorded in real environments often contains **room reverberation and environmental noise**, which reduces speech clarity and intelligibility.

This project aims to build a **hybrid signal processing + neural network pipeline** that can:

- Simulate realistic reverberant environments
- Train a neural model for speech enhancement
- Allow adaptive dereverberation control

The goal is to develop a **research-oriented speech enhancement system** that can later be extended for **real-time speech applications**.

---
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

# System Pipeline

The enhancement system combines **classical DSP processing with a neural enhancement model**.

```

Clean Speech
│
▼
Room Impulse Response (RIR)
│
▼
Reverberant Speech
│
▼
Add Environmental Noise
│
▼
Noisy Reverberant Mixture
│
▼
STFT Feature Extraction
│
▼
CNN Neural Enhancement Model
│
▼
Spectral Masking
│
▼
Inverse STFT
│
▼
Enhanced Speech

```

The model learns to recover cleaner speech from noisy reverberant mixtures.

---

# Model

The neural component uses a **CNN-based spectral enhancement model**.

Key ideas:

- STFT based spectral features
- CNN layers for feature extraction
- Spectral mask prediction
- Reconstruction using inverse STFT

The neural model is trained to map **noisy reverberant speech → enhanced speech**.

---

# Dataset

Training samples are generated **on-the-fly** using the **DNS Challenge datasets**.

Datasets used:

- **VCTK Dataset** — clean speech
- **DNS Noise Dataset** — environmental noise
- **SLR26 Simulated RIR Dataset** — room impulse responses

Since the dataset size is **~50GB+**, the datasets are stored **locally and are not included in this repository**.

---

# Training Configuration

Example training setup:

```

Sample Rate: 48000 Hz
Segment Length: 3 seconds
Batch Size: 4
Epochs: 20
FFT Size: 1024
Hop Length: 480

````

Training command:

```bash
python -m src.train --config config.yaml
````

---

# Training Progress

Training was executed for **20 epochs**.

Log output: <img width="1647" height="550" alt="image" src="https://github.com/user-attachments/assets/a7dacefe-3542-44d2-9020-d9dc38a48ee6" />

```
Epoch 20: avg_loss=0.3567
Best Loss: 0.3548
```

The loss decreased across epochs, indicating that the model is learning the enhancement mapping.

---

# Project Development Phases

**Phase 1 — Dataset Setup**
Downloaded and structured DNS datasets; integrated VCTK speech, noise datasets, and room impulse responses.

**Phase 2 — Signal Processing Pipeline**
Implemented STFT/ISTFT processing, reverberation simulation using RIR convolution, and noise injection with controlled SNR.

**Phase 3 — Neural Model Development**
Implemented a CNN-based speech enhancement network operating on spectral features.

**Phase 4 — Training Pipeline Implementation**
Developed on-the-fly dataset generation, safe audio loading, and full model training loop.

**Phase 5 — Initial Model Training**
Ran initial training experiments and verified convergence behaviour.

**Phase 6 — DSP Verification & Evaluation (Current – In Progress)**
Testing the DSP pipeline, validating enhancement quality, and evaluating model performance on various speech samples.

---

# Current Status

The core codebase and training pipeline are implemented.

However, several steps are still pending:

* Full validation of the **DSP pipeline**
* Extensive testing on different speech samples
* Quantitative evaluation using speech metrics
* Real-time inference experiments

The project is **currently in active development**.

---

# Repository Structure

```
adaptive-dereverb-se
│
├── src
│   ├── data.py
│   ├── dsp.py
│   ├── model.py
│   ├── train.py
│   ├── realtime_app.py
│   ├── infer_file.py
│   └── utils.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/Asmi2911/adaptive-dereverb-se
cd adaptive-dereverb-se
pip install -r requirements.txt
```

---

# Run Training

```bash
python -m src.train --config config.yaml
```

---

# Run Inference

```bash
python -m src.infer_file --config config.yaml --input sample.wav --output enhanced.wav
```

---

# Author

**Asmita Sonavane**
M.S. Computer Engineering
NYU Tandon School of Engineering

Advisor: **Prof. Ivan Selesnick**

---

```

