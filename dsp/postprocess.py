import numpy as np
import scipy.signal as sps
import torch


def postprocess_organ_plugin(
    y_base,
    sr,
    brightness=0.015,
    env_ms=15,
    reverb_mix=0.0,
    noise_amount=0.0,
):
    """
    Plugin-safe post-processing.
    Always processes from immutable y_base.
    """

    # --- Convert to numpy ---
    if isinstance(y_base, torch.Tensor):
        y = y_base.detach().cpu().numpy().squeeze()
    else:
        y = np.asarray(y_base).squeeze()

    y = y.copy()  # CRITICAL: avoid in-place corruption

    # --- Envelope smoothing (RMS) ---
    win = int(sr * env_ms / 1000)
    win = max(3, win | 1)

    rms = np.sqrt(
        sps.convolve(y**2, np.ones(win) / win, mode="same") + 1e-8
    )
    rms /= rms.max() + 1e-8
    y *= rms

    # --- High-pass (remove rumble only) ---
    b, a = sps.butter(1, 25 / (sr / 2), btype="high")
    y = sps.lfilter(b, a, y)

    # --- Brightness ---
    if brightness > 0:
        b, a = sps.butter(1, 3500 / (sr / 2), btype="high")
        y += brightness * sps.lfilter(b, a, y)

    # --- Deterministic noise ---
    if noise_amount > 0:
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(len(y))
        noise = sps.lfilter([1], [1, -0.98], noise)  # pink-ish
        noise *= noise_amount * np.sqrt(np.mean(y**2))
        y += noise

    # --- Simple plate reverb ---
    if reverb_mix > 0:
        delay = int(sr * 0.035)
        y_rev = y.copy()
        for i in range(delay, len(y)):
            y_rev[i] += 0.6 * y_rev[i - delay]
        y = (1 - reverb_mix) * y + reverb_mix * y_rev

    # --- Normalize ---
    peak = np.max(np.abs(y))
    if peak > 0:
        y /= peak

    return y
