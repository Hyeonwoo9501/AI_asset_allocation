"""
Quick Start Example Script
Run a complete pipeline: data loading -> model training -> backtesting
"""

import os
import yaml
import torch
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import MarketDataLoader
from models.transformer_model import TransformerFactorModel
from models.loss_functions import compute_metrics

print("=" * 60)
print("Transformer Factor Model - Quick Start Example")
print("=" * 60)

# 1. Load configuration
print("\n[1/5] Loading configuration...")
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("✓ Configuration loaded")
print(f"  - Sector ETFs: {len(config['data']['sector_etfs'])}")
print(f"  - Macro indicators: {len(config['data']['macro_indicators'])}")
print(f"  - Lookback window: {config['data']['lookback_window']}")

# 2. Load data
print("\n[2/5] Loading market data...")
print("  Note: This may take a few minutes for the first run...")

try:
    data_loader = MarketDataLoader(config)
    data_splits = data_loader.load_all_data()

    print("✓ Data loaded successfully")
    print(f"  - Train samples: {len(data_splits['train'][0])}")
    print(f"  - Val samples: {len(data_splits['val'][0])}")
    print(f"  - Test samples: {len(data_splits['test'][0])}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("\nNote: If you see FRED API errors, you need to set your FRED API key.")
    print("Edit utils/data_loader.py and replace 'YOUR_FRED_API_KEY' with your actual key.")
    print("You can get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    exit(1)

# 3. Initialize model
print("\n[3/5] Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  - Device: {device}")

model = TransformerFactorModel(config).to(device)
n_params = sum(p.numel() for p in model.parameters())

print("✓ Model initialized")
print(f"  - Total parameters: {n_params:,}")
print(f"  - Factor dimension: {config['model']['factor_dim']}")

# 4. Test forward pass
print("\n[4/5] Testing model forward pass...")
X_sector, X_macro, y = data_splits['train']

# Take a small batch
batch_size = 8
X_sector_batch = torch.FloatTensor(X_sector[:batch_size]).to(device)
X_macro_batch = torch.FloatTensor(X_macro[:batch_size]).to(device)
y_batch = torch.FloatTensor(y[:batch_size]).to(device)

with torch.no_grad():
    predictions, factors = model(X_sector_batch, X_macro_batch)

print("✓ Forward pass successful")
print(f"  - Predictions shape: {predictions.shape}")
print(f"  - Factors shape: {factors.shape}")

# Compute metrics
metrics = compute_metrics(predictions, y_batch)
print(f"  - Initial MSE: {metrics['mse']:.6f}")
print(f"  - Initial IC: {metrics['ic']:.4f}")

# 5. Next steps
print("\n[5/5] Next steps:")
print("=" * 60)
print("\n✓ Setup complete! Your model is ready to train.")
print("\nTo start training, run:")
print("  python train.py --config configs/config.yaml")
print("\nTo monitor training with TensorBoard:")
print("  tensorboard --logdir results/logs")
print("\nAfter training, run backtesting:")
print("  python inference.py --checkpoint results/checkpoints/best_model.pt")
print("\n" + "=" * 60)

# Optional: Quick 1-epoch demo
print("\nWould you like to run a quick 1-epoch training demo? (y/n)")
response = input().lower().strip()

if response == 'y':
    print("\n[Demo] Running 1 epoch training...")

    from torch.utils.data import DataLoader, TensorDataset
    from models.loss_functions import CompositeLoss

    # Create data loader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_sector[:100]),  # Small subset for demo
        torch.FloatTensor(X_macro[:100]),
        torch.FloatTensor(y[:100])
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = CompositeLoss(config)

    # Train for 1 epoch
    model.train()
    total_loss = 0

    for X_s, X_m, y_true in train_loader:
        X_s, X_m, y_true = X_s.to(device), X_m.to(device), y_true.to(device)

        optimizer.zero_grad()
        pred, factors = model(X_s, X_m)
        loss, loss_dict = criterion(pred, y_true, model)
        loss.backward()
        optimizer.step()

        total_loss += loss_dict['total']

    avg_loss = total_loss / len(train_loader)
    print(f"✓ Demo training complete - Average loss: {avg_loss:.6f}")
    print("  This is just a quick demo. For full training, run train.py")

print("\nThank you for using Transformer Factor Model!")
