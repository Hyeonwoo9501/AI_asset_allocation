"""
Training Script for Factor Embedding Model (Stage 1)

Train the embedding model to extract meaningful factors from ETF + Macro data
Uses contrastive learning or reconstruction objectives
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models.embedding_model import FactorEmbeddingModel
from utils.data_loader import MarketDataLoader


class EmbeddingLoss(nn.Module):
    """
    Loss function for factor embedding learning

    Strategy: Predict next-period returns from factor embeddings
    (Similar to original model, but cleaner separation)
    """

    def __init__(self, factor_dim: int, n_assets: int):
        super().__init__()
        # Simple linear layer: factors → returns
        self.predictor = nn.Linear(factor_dim, n_assets)

    def forward(self, factors, target_returns):
        """
        Args:
            factors: (batch, factor_dim)
            target_returns: (batch, n_assets)

        Returns:
            loss: scalar
            metrics: dict
        """
        # Predict returns from factors
        pred_returns = self.predictor(factors)

        # MSE loss
        mse_loss = nn.functional.mse_loss(pred_returns, target_returns)

        # IC loss (ranking correlation)
        ic = self._compute_ic(pred_returns, target_returns)
        ic_loss = -ic  # maximize IC

        # Total loss
        total_loss = mse_loss + 0.5 * ic_loss

        metrics = {
            'mse': mse_loss.item(),
            'ic': ic.item(),
            'total': total_loss.item()
        }

        return total_loss, metrics

    def _compute_ic(self, predictions, targets):
        """Compute Information Coefficient (rank correlation)"""
        # Rank predictions and targets
        pred_ranks = torch.argsort(torch.argsort(predictions, dim=1), dim=1).float()
        target_ranks = torch.argsort(torch.argsort(targets, dim=1), dim=1).float()

        # Compute correlation
        pred_centered = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
        target_centered = target_ranks - target_ranks.mean(dim=1, keepdim=True)

        numerator = (pred_centered * target_centered).sum(dim=1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1))

        ic = numerator / (pred_std * target_std + 1e-8)
        return ic.mean()


class EmbeddingTrainer:
    """Trainer for Factor Embedding Model"""

    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device(
            self.config['training']['device']
            if torch.cuda.is_available()
            else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Load data
        print("\n=== Loading Data ===")
        data_loader = MarketDataLoader(self.config)
        self.data_splits = data_loader.load_all_data()

        # Create data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        self.test_loader = self._create_dataloader('test')

        # Initialize model
        print("\n=== Initializing Model ===")
        self.model = FactorEmbeddingModel(self.config).to(self.device)
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Factor dimension: {self.config['model']['factor_dim']}")

        # Initialize loss function
        n_sector = len(self.config['data']['sector_etfs'])
        n_additional = len(self.config['data'].get('additional_etfs', []))
        n_assets = n_sector + n_additional
        self.criterion = EmbeddingLoss(
            factor_dim=self.config['model']['factor_dim'],
            n_assets=n_assets
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create directories
        os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['training']['log_dir'], exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(self.config['training']['log_dir'])

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create PyTorch DataLoader for a data split"""
        X_etf, X_macro, y = self.data_splits[split]

        # Convert to tensors
        X_etf = torch.FloatTensor(X_etf)
        X_macro = torch.FloatTensor(X_macro)
        y = torch.FloatTensor(y)

        # Create dataset
        dataset = TensorDataset(X_etf, X_macro, y)

        # Create dataloader
        batch_size = self.config['training']['batch_size']
        shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return dataloader

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        self.criterion.train()

        epoch_metrics = {'mse': 0.0, 'ic': 0.0, 'total': 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for X_etf, X_macro, y in pbar:
            X_etf, X_macro, y = X_etf.to(self.device), X_macro.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            factors = self.model(X_etf, X_macro)
            loss, metrics = self.criterion(factors, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                self.config['training']['grad_clip']
            )

            self.optimizer.step()

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

            # Update progress bar
            pbar.set_postfix({
                'loss': metrics['total'],
                'mse': metrics['mse'],
                'ic': metrics['ic']
            })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.train_loader)

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set"""
        self.model.eval()
        self.criterion.eval()

        epoch_metrics = {'mse': 0.0, 'ic': 0.0, 'total': 0.0}

        for X_etf, X_macro, y in self.val_loader:
            X_etf, X_macro, y = X_etf.to(self.device), X_macro.to(self.device), y.to(self.device)

            # Forward pass
            factors = self.model(X_etf, X_macro)
            loss, metrics = self.criterion(factors, y)

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.val_loader)

        return epoch_metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            'latest_model.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (val_loss: {self.best_val_loss:.6f})")

    def train(self):
        """Main training loop"""
        print("\n=== Starting Training ===")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log to tensorboard
            for key in train_metrics:
                self.writer.add_scalar(f'train/{key}', train_metrics[key], epoch)
                self.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)

            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Print progress
            if epoch % self.config['training']['log_interval'] == 0:
                print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
                print(f"  Train - Loss: {train_metrics['total']:.6f}, "
                      f"MSE: {train_metrics['mse']:.6f}, IC: {train_metrics['ic']:.4f}")
                print(f"  Val   - Loss: {val_metrics['total']:.6f}, "
                      f"MSE: {val_metrics['mse']:.6f}, IC: {val_metrics['ic']:.4f}")

            # Save checkpoint
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % self.config['training']['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print("\n=== Training Complete ===")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        self.writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Factor Embedding Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Train
    trainer = EmbeddingTrainer(args.config)
    trainer.train()

    print("\n" + "="*60)
    print("Training complete! Next steps:")
    print("1. Run factor analysis:")
    print("   python factor_portfolio.py")
    print("2. Check TensorBoard logs:")
    print("   tensorboard --logdir results/logs")
    print("="*60)
