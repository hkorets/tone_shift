#!/usr/bin/env python3
import subprocess
import os

INPUT = "noised_hrystia_piano.wav"
OUTPUT_DIR = "models/denoise/presentation_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Creating demo files...")

subprocess.run(["cp", INPUT, f"{OUTPUT_DIR}/1_original_noisy.wav"])

subprocess.run(["python3", "models/denoise/test.py", INPUT, 
                f"{OUTPUT_DIR}/3_deep_learning.wav"])

print(f"Demo files ready in {OUTPUT_DIR}/")
