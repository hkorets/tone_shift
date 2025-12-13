import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import sys
import os

class LightweightDenoiser(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, base_channels, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*4, 15, padding=7),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels*2, 15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels, 15, stride=2, padding=7, output_padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*2, 1, 15, stride=2, padding=7, output_padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottleneck(e3)
        
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1

def denoise_audio(model, input_path, output_path, device="mps"):
    """Denoise an audio file"""
    
    print(f"\n{'='*50}")
    print(f"Denoising: {input_path}")
    print(f"{'='*50}")
    
    # Load audio
    audio, sr = sf.read(input_path)
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio)/sr:.2f} seconds")
    
    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
        print("Converted to mono")
    
    # Normalize
    audio_max = np.max(np.abs(audio))
    audio_normalized = audio / (audio_max + 1e-8)
    
    # Process in chunks if audio is long
    chunk_size = sr * 10  # 10 seconds
    
    if len(audio_normalized) > chunk_size:
        print(f"Processing in chunks (long audio)...")
        
        denoised_chunks = []
        overlap = chunk_size // 4
        
        for i in range(0, len(audio_normalized), chunk_size - overlap):
            chunk = audio_normalized[i:i + chunk_size]
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Convert to tensor
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            
            # Denoise
            with torch.no_grad():
                denoised_chunk = model(chunk_tensor)
            
            denoised_chunk = denoised_chunk.squeeze().cpu().numpy()
            
            # Handle overlap
            if i > 0 and len(denoised_chunks) > 0:
                # Crossfade
                fade_len = min(overlap, len(denoised_chunk))
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                
                denoised_chunks[-1][-fade_len:] = (
                    denoised_chunks[-1][-fade_len:] * fade_out + 
                    denoised_chunk[:fade_len] * fade_in
                )
                denoised_chunks.append(denoised_chunk[fade_len:])
            else:
                denoised_chunks.append(denoised_chunk)
        
        # Concatenate
        denoised = np.concatenate(denoised_chunks)[:len(audio_normalized)]
    
    else:
        # Short audio - process all at once
        print("Processing...")
        audio_tensor = torch.FloatTensor(audio_normalized).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            denoised_tensor = model(audio_tensor)
        
        denoised = denoised_tensor.squeeze().cpu().numpy()
    
    # Denormalize
    denoised = denoised * audio_max
    
    # Ensure same length
    denoised = denoised[:len(audio)]
    
    # Save
    sf.write(output_path, denoised, sr)
    
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*50}\n")
    
    return denoised, sr

def main():
    print("\n" + "="*70)
    print(" " * 20 + "AUDIO DENOISER")
    print("="*70)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_model.py input.wav [output.wav] [model.pt]")
        print("\nExamples:")
        print("  python test_model.py noisy.wav")
        print("  python test_model.py noisy.wav clean.wav")
        print("  python test_model.py noisy.wav clean.wav checkpoints/best.pt")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"denoised_{os.path.basename(input_path)}"
    model_path = sys.argv[3] if len(sys.argv) > 3 else "models/denoise/checkpoints/best.pt"
    
    # Check input file exists
    if not os.path.exists(input_path):
        print(f"\n❌ ERROR: Input file not found: {input_path}")
        return
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model file not found: {model_path}")
        print("Did you train the model? Run lightweight_training.py first.")
        return
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = LightweightDenoiser(base_channels=32).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # Denoise
    denoised, sr = denoise_audio(model, input_path, output_path, device)
    
    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "COMPLETE!")
    print("="*70)
    print(f"\nOriginal: {input_path}")
    print(f"Denoised: {output_path}")
    print("\nListen to both files to compare!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()