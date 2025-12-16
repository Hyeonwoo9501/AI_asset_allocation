"""
Training Script for Transformer Factor Model
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models.transformer_model import TransformerFactorModel
from models.loss_functions import CompositeLoss, compute_metrics
from utils.data_loader import MarketDataLoader


class Trainer:
    """Trainer for Transformer Factor Model"""

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
        self.model = TransformerFactorModel(self.config).to(self.device)
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Initialize loss function
        self.criterion = CompositeLoss(self.config)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

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
        X_sector, X_macro, y = self.data_splits[split]

        # Convert to tensors
        X_sector = torch.FloatTensor(X_sector)
        X_macro = torch.FloatTensor(X_macro)
        y = torch.FloatTensor(y)

        # Create dataset
        dataset = TensorDataset(X_sector, X_macro, y)

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

    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler'].lower()
        epochs = self.config['training']['epochs']

        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs
            )
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_metrics = {'mse': 0, 'ic': 0, 'sharpe': 0, 'l1': 0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for X_sector, X_macro, y in pbar:
            # Move to device
            X_sector = X_sector.to(self.device)
            X_macro = X_macro.to(self.device)
            y = y.to(self.device)

            # Forward pass
            predictions, factors = self.model(X_sector, X_macro)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, y, self.model)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss_dict['total']
            for key in ['mse', 'ic', 'sharpe', 'l1']:
                total_metrics[key] += loss_dict[key]
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'ic': loss_dict['ic']
            })

        # Average metrics
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

        return {'loss': avg_loss, **avg_metrics}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        total_metrics = {'mse': 0, 'ic': 0, 'sharpe': 0, 'l1': 0}
        n_batches = 0

        all_predictions = []
        all_targets = []

        for X_sector, X_macro, y in dataloader:
            # Move to device
            X_sector = X_sector.to(self.device)
            X_macro = X_macro.to(self.device)
            y = y.to(self.device)

            # Forward pass
            predictions, factors = self.model(X_sector, X_macro)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, y, self.model)

            # Update metrics
            total_loss += loss_dict['total']
            for key in ['mse', 'ic', 'sharpe', 'l1']:
                total_metrics[key] += loss_dict[key]
            n_batches += 1

            # Store predictions and targets
            all_predictions.append(predictions)
            all_targets.append(y)

        # Average metrics
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

        # Compute additional evaluation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        eval_metrics = compute_metrics(all_predictions, all_targets)

        return {'loss': avg_loss, **avg_metrics, **eval_metrics}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.6f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        print("\n=== Starting Training ===")

        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate(self.val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('IC/train', train_metrics['ic'], epoch)
            self.writer.add_scalar('IC/val', val_metrics['ic'], epoch)
            self.writer.add_scalar('Sharpe/train', train_metrics['sharpe'], epoch)
            self.writer.add_scalar('Sharpe/val', val_metrics['sharpe'], epoch)

            # Print metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.6f}, IC: {train_metrics['ic']:.4f}, Sharpe: {train_metrics['sharpe']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.6f}, IC: {val_metrics['ic']:.4f}, Sharpe: {val_metrics['sharpe']:.4f}")

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Final evaluation on test set
        print("\n=== Final Evaluation on Test Set ===")
        test_metrics = self.validate(self.test_loader)
        print(f"Test - Loss: {test_metrics['loss']:.6f}")
        print(f"Test - IC: {test_metrics['ic']:.4f}")
        print(f"Test - Sharpe: {test_metrics['sharpe']:.4f}")
        print(f"Test - Portfolio Return: {test_metrics['portfolio_return']:.6f}")

        self.writer.close()
        print("\nTraining complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Transformer Factor Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()
