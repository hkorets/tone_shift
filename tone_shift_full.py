from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from features.feature_extraction import extract_features
from models.denoise.test import denoise_audio, LightweightDenoiser

from synth.synthesis import synthesize_from_params, make_wav


class TimbreDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        mlp_hidden: int = 256,
        harmonics_dim: int = 60,
        noise_dim: int = 65,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.harmonics_head = nn.Linear(mlp_hidden, harmonics_dim)
        self.noise_head = nn.Linear(mlp_hidden, noise_dim)
        self.gain_head = nn.Linear(mlp_hidden, 1)

    def forward(self, x, h0=None):
        gru_out, h_n = self.gru(x, h0)
        z = self.mlp(gru_out)

        harmonics = self.harmonics_head(z)
        noise = self.noise_head(z)
        gain = self.gain_head(z)

        harmonics = F.softplus(harmonics)
        harmonics_sum = harmonics.sum(dim=-1, keepdim=True) + 1e-8
        harmonics = harmonics / harmonics_sum

        noise = F.softplus(noise)
        gain = torch.sigmoid(gain)

        return harmonics, noise, gain, h_n


def load_decoder(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    K_H: int = 60,
    K_N: int = 65,
) -> TimbreDecoder:
    device = torch.device(device) if not isinstance(device, torch.device) else device

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    decoder = TimbreDecoder(in_dim=3, harmonics_dim=K_H, noise_dim=K_N).to(device)
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()
    return decoder


@torch.no_grad()
def run_inference(
    decoder: TimbreDecoder,
    source_wav: Union[str, Path],
    prepare_features_for_decoder,
    SR: int = 16000,
    HOP_LENGTH: int = 512,
    noise_scale: float = 0.0,
    device: Union[str, torch.device] = "cpu",
    out_wav_path: Optional[Union[str, Path]] = "test_synth.wav",
) -> torch.Tensor:

    device = torch.device(device) if not isinstance(device, torch.device) else device
    decoder = decoder.to(device).eval()

    FRAME_RATE = SR / HOP_LENGTH

    feats_t, f0_t, audio_t, frame_rate = prepare_features_for_decoder(str(source_wav))

    feats_t = feats_t.to(device)
    f0_t = f0_t.to(device)

    harmonics, noise, gain, _ = decoder(feats_t)

    params: Dict[str, Any] = {
        "harmonic_amps": harmonics,
        "noise_amps": noise,
        "gain": gain,
        "optional": {},
    }

    y_hat = synthesize_from_params(
        params,
        f0_t,
        sr=SR,
        frame_rate=FRAME_RATE,
        noise_scale=noise_scale,
    )

    if out_wav_path is not None:
        make_wav(y_hat, SR) 

    return y_hat

SR = 16000
HOP_LENGTH = 512
FRAME_RATE = SR / HOP_LENGTH

def prepare_features_for_decoder(wav_path: str):
    out = extract_features(wav_path)

    if len(out) == 4:
        f0, loudness, onset, y = out
    elif len(out) == 5:
        f0, loudness, onset, y, _ = out   
    else:
        raise RuntimeError(f"Unexpected number of outputs from extract_features: {len(out)}")

    f0 = np.asarray(f0, dtype=np.float32)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0_hz = np.clip(f0, 1.0, 3000.0).astype(np.float32)
    f0_log = np.log2(f0_hz / 440.0).astype(np.float32)

    loudness = np.asarray(loudness, dtype=np.float32)
    onset = np.asarray(onset, dtype=np.float32)

    loudness = np.nan_to_num(loudness, nan=0.0, posinf=0.0, neginf=0.0)
    onset = np.nan_to_num(onset, nan=0.0, posinf=0.0, neginf=0.0)

    T = min(len(f0_log), len(loudness), len(onset))
    f0_hz = f0_hz[:T]
    f0_log = f0_log[:T]
    loudness = loudness[:T]
    onset = onset[:T]

    feats = np.stack([f0_log, loudness, onset], axis=-1).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    feats_t = torch.from_numpy(feats).unsqueeze(0)
    f0_t = torch.from_numpy(f0_hz).unsqueeze(0)
    audio_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)

    frame_rate = FRAME_RATE
    return feats_t, f0_t, audio_t, frame_rate


ckpt_path = r"C:\Users\Admin\Desktop\tone_shift\checkpoints\decoder_organ_ddsp.pth"
source_wav = r"C:\Users\Admin\Desktop\tone_shift\document_5305266375659393036_cut_0_80000.wav"


# -----------------------------------------------------------------------
# timbre transfer inference function
# -----------------------------------------------------------------------

def infer_from_wav(
    wav_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    K_H: int = 60,
    K_N: int = 65,
    SR: int = 16000,
    HOP_LENGTH: int = 512,
    noise_scale: float = 0.0,
    save_wav: bool = False,
    out_wav_path: Union[str, Path] = "test_synth.wav",
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    decoder = load_decoder(
        ckpt_path=ckpt_path,
        device=device,
        K_H=K_H,
        K_N=K_N,
    )

    y_hat = run_inference(
        decoder=decoder,
        source_wav=wav_path,
        prepare_features_for_decoder=prepare_features_for_decoder,
        SR=SR,
        HOP_LENGTH=HOP_LENGTH,
        noise_scale=noise_scale,
        device=device,
        out_wav_path=out_wav_path if save_wav else None,
    )

    return y_hat

#-----------------------------------------------------------------------
# denoising inference function
#-----------------------------------------------------------------------
def denoise_wav(input_path, output_path, model_path = "models/denoise/checkpoints/best.pt"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LightweightDenoiser(base_channels=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    denoised, sr = denoise_audio(model, input_path, output_path, device)
    print("denoised")

    
#-----------------------------------------------------------------------
# pipeline
#-----------------------------------------------------------------------


# y_denoised = denoise_wav(
#     input_path=r"C:\Users\Admin\Desktop\tone_shift\document_5305266375659393036_cut_0_80000.wav",
#     output_path=r"C:\Users\Admin\Desktop\tone_shift\denoised_document_5305266375659393036_cut_0_80000.wav",
#     model_path = r"C:\Users\Admin\Desktop\tone_shift\models\denoise\checkpoints\best.pt")

# y_hat = infer_from_wav(
#     wav_path=r"C:\Users\Admin\Desktop\tone_shift\document_5305266375659393036_cut_0_80000.wav",
#     ckpt_path=r"C:\Users\Admin\Desktop\tone_shift\checkpoints\decoder_organ_ddsp.pth",
#     save_wav=True,
# )
