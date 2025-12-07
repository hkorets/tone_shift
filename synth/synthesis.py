import torch
import torch.nn.functional as F
import math
import soundfile as sf


def synthesize_from_params(
    params,
    f0,
    sr=16000,
    frame_rate=100,
    noise_scale=1.0,       # <--- NEW
    return_parts=False,    # <--- NEW
):
    """
    params: dict
      harmonic_amps: (B, T, K_h)
      noise_amps:    (B, T, K_n)
      gain:          (B, T, 1)
      optional: maybe filter_coeffs, transient (B, T, 1)

    f0: (B, T) fundamental frequency in Hz
    """
    B, T, K_h = params["harmonic_amps"].shape
    # print(B, T, K_h)   # can comment out now
    _, _, K_n = params["noise_amps"].shape
    n_samples = int(T * sr / frame_rate)
    device = params["harmonic_amps"].device

    def interp_param(p):
        B, T, X = p.shape
        p2 = p.permute(0, 2, 1)
        p2 = F.interpolate(p2, size=n_samples, mode="linear", align_corners=False)
        p3 = p2.permute(0, 2, 1)
        return p3

    harm_amp_s = interp_param(params["harmonic_amps"])
    noise_amp_s = interp_param(params["noise_amps"])
    gain_s = interp_param(params["gain"]).squeeze(-1)
    f0_s = (
        F.interpolate(
            f0.unsqueeze(-1).permute(0, 2, 1),
            size=n_samples,
            mode="linear",
            align_corners=False,
        )
        .permute(0, 2, 1)
        .squeeze(-1)
    )

    if "transient" in params.get("optional", {}):
        transient_s = interp_param(params["optional"]["transient"]).squeeze(-1)
    else:
        transient_s = None

    dt = 1.0 / sr
    phase_f0 = torch.cumsum(f0_s * (2 * math.pi * dt), dim=1)  # [B, n_samples]
    harmonics_idx = torch.arange(1, K_h + 1, device=device).float()  # [K_h]
    phases = phase_f0.unsqueeze(-1) * harmonics_idx                 # [B, n_samples, K_h]
    sinusoids = torch.sin(phases)                                   # [B, n_samples, K_h]
    y_harm = (harm_amp_s * sinusoids).sum(dim=-1)                   # [B, n_samples]

    noise = torch.randn_like(y_harm)
    if K_n == 1:
        y_noise = noise * noise_amp_s.squeeze(-1) * noise_scale
    else:
        # average over noise bands, then scale
        y_noise = noise * noise_amp_s.mean(dim=-1) * noise_scale

    y = y_harm + y_noise
    if transient_s is not None:
        y = y + transient_s
    y = y * gain_s

    # optional filter_coeffs not used

    y = torch.clamp(y, -1.0, 1.0)

    if return_parts:
        return y, y_harm, y_noise

    return y


def make_wav(y, sr):
    waveform = y[0].detach().cpu().numpy()
    waveform = waveform / (abs(waveform).max() + 1e-6)
    sf.write("test_synth.wav", waveform, sr)
    print("✅ Audio saved to test_synth.wav")
