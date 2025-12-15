# =========================
# Path & imports (CRITICAL)
# =========================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import tempfile
import soundfile as sf
import torch

from tone_shift_full import infer_from_wav
from dsp.postprocess import postprocess_organ_plugin


# =========================
# Config
# =========================


SR = 16000
CKPT_PATH = ROOT / "checkpoints" / "decoder_organ_ddsp.pth"

st.set_page_config(
    page_title="ToneShift – Timbre Transfer Demo",
    layout="centered",
)

st.title("🎹 ToneShift – Demo")

# =========================
# Upload section
# =========================
uploaded = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded.read())
        input_wav_path = f.name

    st.audio(input_wav_path)

    # Run model ONCE
    if st.button("Run Timbre Transfer"):
        with st.spinner("Running timbre transfer model..."):
            y_hat = infer_from_wav(
                wav_path=input_wav_path,
                ckpt_path=CKPT_PATH,
                save_wav=False,
            )

        # IMMUTABLE BASE BUFFER
        st.session_state["y_base"] = y_hat
        st.success("Timbre transfer complete!")


# =========================
# Post-processing section
# =========================
if "y_base" in st.session_state:
    st.subheader("🎛 Post-processing")

    brightness = st.slider(
        "Brightness",
        min_value=0.0,
        max_value=0.05,
        value=0.015,
        step=0.001,
    )

    env_ms = st.slider(
        "Envelope smoothing (ms)",
        min_value=5,
        max_value=40,
        value=15,
    )

    noise_amount = st.slider(
        "Air / Noise",
        min_value=0.0,
        max_value=0.02,
        value=0.0,
        step=0.001,
    )

    reverb_mix = st.slider(
        "Reverb",
        min_value=0.0,
        max_value=0.3,
        value=0.0,
        step=0.01,
    )

    # ALWAYS recompute from y_base (NEVER cumulative)
    y_post = postprocess_organ_plugin(
        y_base=st.session_state["y_base"],
        sr=SR,
        brightness=brightness,
        env_ms=env_ms,
        noise_amount=noise_amount,
        reverb_mix=reverb_mix,
    )

    # Write temp output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, y_post, SR)
        out_path = f.name

    st.audio(out_path)

    with open(out_path, "rb") as f:
        st.download_button(
            "⬇ Download processed WAV",
            f,
            file_name="timbre_transfer_output.wav",
            mime="audio/wav",
        )

    # Reset button (plugin-style)
    if st.button("Reset Post-processing"):
        st.rerun()
