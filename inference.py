"""
Inference and Backtesting Script for Transformer Factor Model
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from models.transformer_model import TransformerFactorModel
from utils.data_loader import MarketDataLoader


class PortfolioBacktester:
    """Backtest trading strategy based on model predictions"""

    def __init__(self, config_path: str, checkpoint_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Load model
        print("\n=== Loading Model ===")
        self.model = TransformerFactorModel(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        # Load data
        print("\n=== Loading Data ===")
        data_loader = MarketDataLoader(self.config)
        self.data_splits = data_loader.load_all_data()

        # Portfolio configuration
        self.top_k = self.config['portfolio']['top_k']
        self.rebalance_freq = self.config['portfolio']['rebalance_freq']
        self.transaction_cost = self.config['portfolio']['transaction_cost']

        # Asset names
        self.asset_names = self.config['data']['sector_etfs']

    @torch.no_grad()
    def predict(self, X_sector: np.ndarray, X_macro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data
        Returns:
            predictions: (n_samples, n_assets)
            factors: (n_samples, factor_dim)
        """
        # Convert to tensors
        X_sector = torch.FloatTensor(X_sector).to(self.device)
        X_macro = torch.FloatTensor(X_macro).to(self.device)

        # Predict
        predictions, factors = self.model(X_sector, X_macro)

        # Convert to numpy
        predictions = predictions.cpu().numpy()
        factors = factors.cpu().numpy()

        return predictions, factors

    def construct_portfolio(self, predictions: np.ndarray) -> np.ndarray:
        """
        Construct portfolio weights based on predictions
        Long top-k assets with equal weights
        Args:
            predictions: (n_samples, n_assets)
        Returns:
            weights: (n_samples, n_assets)
        """
        n_samples, n_assets = predictions.shape
        weights = np.zeros_like(predictions)

        for i in range(n_samples):
            # Get top-k assets
            top_k_indices = np.argsort(predictions[i])[-self.top_k:]

            # Equal weight
            weights[i, top_k_indices] = 1.0 / self.top_k

        return weights

    def backtest(self, split: str = 'test') -> Dict:
        """
        Run backtest on specified data split
        Args:
            split: 'train', 'val', or 'test'
        Returns:
            backtest_results: dict containing performance metrics
        """
        print(f"\n=== Running Backtest on {split.upper()} set ===")

        # Get data
        X_sector, X_macro, y = self.data_splits[split]

        # Get predictions
        predictions, factors = self.predict(X_sector, X_macro)

        # Construct portfolio weights
        weights = self.construct_portfolio(predictions)

        # Calculate portfolio returns (without rebalancing constraint)
        portfolio_returns_daily = (weights * y).sum(axis=1)

        # Apply rebalancing (hold positions for rebalance_freq days)
        portfolio_returns_rebalanced = []
        portfolio_weights_history = []

        for i in range(0, len(predictions), self.rebalance_freq):
            # Rebalance portfolio
            current_weights = weights[i]
            portfolio_weights_history.append(current_weights)

            # Calculate returns for next rebalance_freq days
            for j in range(self.rebalance_freq):
                idx = i + j
                if idx >= len(y):
                    break

                # Portfolio return
                port_return = (current_weights * y[idx]).sum()

                # Subtract transaction cost (only on rebalance day)
                if j == 0 and i > 0:
                    turnover = np.abs(current_weights - weights[i - self.rebalance_freq]).sum()
                    port_return -= turnover * self.transaction_cost

                portfolio_returns_rebalanced.append(port_return)

        portfolio_returns = np.array(portfolio_returns_rebalanced)

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio_returns, y)

        # Store results
        results = {
            'predictions': predictions,
            'factors': factors,
            'weights': weights,
            'weights_history': portfolio_weights_history,
            'portfolio_returns': portfolio_returns,
            'metrics': metrics
        }

        # Print metrics
        print("\n=== Backtest Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        return results

    def calculate_performance_metrics(
        self,
        portfolio_returns: np.ndarray,
        asset_returns: np.ndarray
    ) -> Dict:
        """Calculate performance metrics"""

        # Cumulative returns
        cumulative_return = (1 + portfolio_returns).prod() - 1

        # Annualized return (assuming 252 trading days)
        n_days = len(portfolio_returns)
        annualized_return = (1 + cumulative_return) ** (252 / n_days) - 1

        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (portfolio_returns > 0).mean()

        # Information Coefficient (average IC over time)
        # Compare each day's prediction vs actual
        # This is simplified - you may want more sophisticated IC calculation
        ic_list = []
        equal_weight_returns = asset_returns.mean(axis=1)
        benchmark_cumulative = (1 + equal_weight_returns).prod() - 1

        metrics = {
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'benchmark_return': benchmark_cumulative
        }

        return metrics

    def plot_results(self, results: Dict, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Cumulative returns
        portfolio_returns = results['portfolio_returns']
        cumulative_returns = (1 + portfolio_returns).cumprod()

        axes[0, 0].plot(cumulative_returns, label='Portfolio', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Portfolio weights over time
        weights_history = np.array(results['weights_history'])
        rebalance_dates = np.arange(0, len(results['portfolio_returns']), self.rebalance_freq)

        for i, asset_name in enumerate(self.asset_names):
            axes[1, 0].plot(
                rebalance_dates[:len(weights_history)],
                weights_history[:, i],
                label=asset_name,
                alpha=0.7
            )

        axes[1, 0].set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Return distribution
        axes[1, 1].hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(portfolio_returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1, 1].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")

        plt.show()

    def analyze_factors(self, results: Dict, save_path: str = None):
        """Analyze extracted factors"""
        factors = results['factors']

        # Factor correlation
        factor_corr = np.corrcoef(factors.T)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Factor correlation heatmap
        sns.heatmap(
            factor_corr,
            cmap='coolwarm',
            center=0,
            square=True,
            ax=axes[0],
            cbar_kws={'label': 'Correlation'}
        )
        axes[0].set_title('Factor Correlation Matrix', fontsize=14, fontweight='bold')

        # 2. Factor time series (first 5 factors)
        for i in range(min(5, factors.shape[1])):
            axes[1].plot(factors[:, i], label=f'Factor {i+1}', alpha=0.7)

        axes[1].set_title('Factor Time Series (First 5)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Factor Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFactor analysis plot saved to {save_path}")

        plt.show()

    def get_factor_weights_importance(self) -> pd.DataFrame:
        """Get factor-to-asset weights for interpretability"""
        weights = self.model.get_factor_weights()  # (n_assets, factor_dim)
        weights = weights.cpu().numpy()

        # Create DataFrame
        df = pd.DataFrame(
            weights.T,
            columns=self.asset_names,
            index=[f'Factor_{i+1}' for i in range(weights.shape[1])]
        )

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Transformer Factor Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Data split to backtest on'
    )
    args = parser.parse_args()

    # Create backtester
    backtester = PortfolioBacktester(args.config, args.checkpoint)

    # Run backtest
    results = backtester.backtest(split=args.split)

    # Plot results
    plot_path = f'results/figures/backtest_{args.split}.png'
    os.makedirs('results/figures', exist_ok=True)
    backtester.plot_results(results, save_path=plot_path)

    # Analyze factors
    factor_path = f'results/figures/factors_{args.split}.png'
    backtester.analyze_factors(results, save_path=factor_path)

    # Get factor weights
    factor_weights = backtester.get_factor_weights_importance()
    print("\n=== Factor Weights (Θ) ===")
    print(factor_weights)

    # Save results
    results_path = f'results/backtest_{args.split}_results.npz'
    np.savez(
        results_path,
        predictions=results['predictions'],
        factors=results['factors'],
        weights=results['weights'],
        portfolio_returns=results['portfolio_returns']
    )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
