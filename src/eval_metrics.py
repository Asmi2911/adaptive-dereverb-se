from __future__ import annotations

import argparse

import numpy as np

from .utils import read_audio


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = alpha * reference
    noise = estimate - target
    ratio = (np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)
    return float(10.0 * np.log10(ratio + 1e-12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--estimate", type=str, required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = read_audio(args.clean, args.sample_rate)
    estimate = read_audio(args.estimate, args.sample_rate)
    min_len = min(len(clean), len(estimate))
    clean = clean[:min_len]
    estimate = estimate[:min_len]

    print(f"SI-SDR: {si_sdr(clean, estimate):.3f} dB")

    try:
        from pesq import pesq
        print(f"PESQ: {pesq(args.sample_rate, clean, estimate, 'wb'):.3f}")
    except Exception as e:
        print(f"PESQ unavailable: {e}")

    try:
        from pystoi import stoi
        print(f"STOI: {stoi(clean, estimate, args.sample_rate, extended=False):.3f}")
    except Exception as e:
        print(f"STOI unavailable: {e}")


if __name__ == "__main__":
    main()
