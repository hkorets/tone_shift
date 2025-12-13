
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LightweightDenoiser(nn.Module):
    """Smaller, faster model that won't crash your laptop"""
    def __init__(self, base_channels=32):
        super().__init__()
        
        # Encoder - only 3 levels instead of 4
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
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*4, 15, padding=7),
            nn.ReLU()
        )
        
        # Decoder
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



class StaticPairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        
        clean, _ = sf.read(clean_path)
        noisy, _ = sf.read(noisy_path)
        
        if clean.ndim > 1:
            clean = np.mean(clean, axis=-1)
        if noisy.ndim > 1:
            noisy = np.mean(noisy, axis=-1)
        
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        max_val = max(np.max(np.abs(clean)), np.max(np.abs(noisy)))
        if max_val > 0:
            clean = clean / max_val
            noisy = noisy / max_val
        
        return torch.FloatTensor(noisy).unsqueeze(0), torch.FloatTensor(clean).unsqueeze(0)



def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for noisy, clean in tqdm(dataloader, desc="Training"):
        noisy, clean = noisy.to(device), clean.to(device)
        
        pred = model(noisy)
        loss = criterion(pred, clean)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating"):
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)



def main():
    print("="*70)
    print("LIGHTWEIGHT TRAINING - Won't Crash Your Laptop!")
    print("="*70)
    
    OUTPUT_DIR = "../../data/denoise/output_pairs"
    BATCH_SIZE = 16     
    NUM_EPOCHS = 15     
    LEARNING_RATE = 5e-4
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    clean_dir = os.path.join(OUTPUT_DIR, "clean")
    noisy_dir = os.path.join(OUTPUT_DIR, "noisy")
    
    pairs = []
    for clean_file in glob.glob(os.path.join(clean_dir, "*.wav")):
        filename = os.path.basename(clean_file)
        noisy_file = os.path.join(noisy_dir, filename)
        if os.path.exists(noisy_file):
            pairs.append((clean_file, noisy_file))
    
    print(f"Found {len(pairs)} pairs")
    
    if len(pairs) == 0:
        print("ERROR: No pairs found! Run preprocessing first.")
        return
    
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(StaticPairsDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(StaticPairsDataset(val_pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create LIGHTWEIGHT model
    model = LightweightDenoiser(base_channels=32).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training loop
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, 'checkpoints/last.pt')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, 'checkpoints/best.pt')
            print("✓ New best model!")
        else:
            patience_counter += 1
            print(f"No improvement: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping!")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print("Model saved: checkpoints/best.pt")


if __name__ == "__main__":
    main()