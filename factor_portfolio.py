"""
Stage 2: Factor Analysis and Portfolio Construction

Given learned factor embeddings, this script:
1. Extracts factors from historical data
2. Analyzes factor-return relationship (linear regression)
3. Computes factor returns and volatility
4. Filters out high-volatility, low-return factors
5. Constructs optimal portfolio using selected factors
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import yaml


class FactorAnalyzer:
    """Analyze factors extracted from embedding model"""

    def __init__(self, factors: np.ndarray, returns: np.ndarray):
        """
        Args:
            factors: (T, factor_dim) - factor time series
            returns: (T, n_assets) - asset returns time series
        """
        self.factors = factors  # (T, K)
        self.returns = returns  # (T, N)
        self.T, self.factor_dim = factors.shape
        self.n_assets = returns.shape[1]

        # Will be computed
        self.beta = None  # (N, K) - factor loadings
        self.factor_returns = None  # (K,) - average return per factor
        self.factor_volatility = None  # (K,) - volatility per factor
        self.factor_sharpe = None  # (K,) - sharpe ratio per factor

    def estimate_factor_loadings(self):
        """
        Estimate beta: r_t = beta @ f_t + epsilon
        using linear regression for each asset
        """
        print("\n=== Estimating Factor Loadings ===")

        beta = np.zeros((self.n_assets, self.factor_dim))

        for i in range(self.n_assets):
            # Linear regression: returns[i] = beta[i] @ factors
            reg = LinearRegression(fit_intercept=True)
            reg.fit(self.factors, self.returns[:, i])
            beta[i] = reg.coef_

        self.beta = beta
        print(f"Beta shape: {beta.shape}")
        print(f"Beta mean: {np.abs(beta).mean():.4f}")

        return beta

    def compute_factor_metrics(self):
        """
        Compute factor-level metrics:
        - Factor returns: contribution to asset returns
        - Factor volatility: std of factor values
        - Factor sharpe: return / volatility
        """
        print("\n=== Computing Factor Metrics ===")

        if self.beta is None:
            self.estimate_factor_loadings()

        # Factor returns: how much each factor contributes to returns
        # Use correlation between factor and portfolio it would create
        factor_returns = np.zeros(self.factor_dim)

        for k in range(self.factor_dim):
            # Portfolio that loads only on factor k
            factor_portfolio = self.beta[:, k]
            factor_portfolio = factor_portfolio / (np.abs(factor_portfolio).sum() + 1e-8)

            # Returns of this portfolio
            portfolio_returns = self.returns @ factor_portfolio

            # Correlation with factor
            factor_returns[k] = np.corrcoef(self.factors[:, k], portfolio_returns)[0, 1]

        # Factor volatility: std of factor values over time
        factor_volatility = self.factors.std(axis=0)

        # Factor sharpe: return / volatility
        factor_sharpe = factor_returns / (factor_volatility + 1e-8)

        self.factor_returns = factor_returns
        self.factor_volatility = factor_volatility
        self.factor_sharpe = factor_sharpe

        print(f"Factor returns - mean: {factor_returns.mean():.4f}, std: {factor_returns.std():.4f}")
        print(f"Factor volatility - mean: {factor_volatility.mean():.4f}, std: {factor_volatility.std():.4f}")
        print(f"Factor sharpe - mean: {factor_sharpe.mean():.4f}, std: {factor_sharpe.std():.4f}")

        return {
            'returns': factor_returns,
            'volatility': factor_volatility,
            'sharpe': factor_sharpe
        }

    def select_factors(
        self,
        min_return: float = 0.0,
        max_volatility: float = None,
        min_sharpe: float = 0.0,
        top_k_pct: float = 0.7
    ) -> np.ndarray:
        """
        Select good factors based on criteria

        Args:
            min_return: minimum factor return
            max_volatility: maximum factor volatility
            min_sharpe: minimum sharpe ratio
            top_k_pct: keep top k% of factors by sharpe

        Returns:
            selected_indices: indices of selected factors
        """
        print("\n=== Selecting Factors ===")

        if self.factor_sharpe is None:
            self.compute_factor_metrics()

        # Criteria
        mask = np.ones(self.factor_dim, dtype=bool)

        # Positive return
        if min_return is not None:
            mask &= (self.factor_returns >= min_return)
            print(f"After min_return filter: {mask.sum()}/{self.factor_dim} factors")

        # Low volatility
        if max_volatility is not None:
            mask &= (self.factor_volatility <= max_volatility)
            print(f"After max_volatility filter: {mask.sum()}/{self.factor_dim} factors")

        # Sharpe ratio
        if min_sharpe is not None:
            mask &= (self.factor_sharpe >= min_sharpe)
            print(f"After min_sharpe filter: {mask.sum()}/{self.factor_dim} factors")

        # Top K%
        if top_k_pct is not None:
            n_keep = int(self.factor_dim * top_k_pct)
            top_k_indices = np.argsort(self.factor_sharpe)[-n_keep:]
            top_k_mask = np.zeros(self.factor_dim, dtype=bool)
            top_k_mask[top_k_indices] = True
            mask &= top_k_mask
            print(f"After top_{top_k_pct*100:.0f}% filter: {mask.sum()}/{self.factor_dim} factors")

        selected_indices = np.where(mask)[0]

        print(f"\nFinal: {len(selected_indices)} factors selected")

        return selected_indices

    def visualize_factors(self, save_path: str = None):
        """Visualize factor analysis results"""

        if self.factor_sharpe is None:
            self.compute_factor_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Factor returns distribution
        axes[0, 0].hist(self.factor_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
        axes[0, 0].set_xlabel('Factor Return')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Factor Return Distribution')
        axes[0, 0].legend()

        # 2. Factor volatility distribution
        axes[0, 1].hist(self.factor_volatility, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Factor Volatility')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Factor Volatility Distribution')

        # 3. Factor sharpe distribution
        axes[1, 0].hist(self.factor_sharpe, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].axvline(0, color='red', linestyle='--', label='Zero')
        axes[1, 0].set_xlabel('Factor Sharpe Ratio')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Factor Sharpe Distribution')
        axes[1, 0].legend()

        # 4. Return vs Volatility scatter
        axes[1, 1].scatter(self.factor_volatility, self.factor_returns, alpha=0.5)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Factor Volatility')
        axes[1, 1].set_ylabel('Factor Return')
        axes[1, 1].set_title('Factor Return vs Volatility')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Factor analysis plot saved to {save_path}")

        plt.close()


class PortfolioOptimizer:
    """
    Portfolio optimization using selected factors

    Strategy: Mean-variance optimization with factor-based risk model
    """

    def __init__(
        self,
        beta: np.ndarray,
        factor_cov: np.ndarray,
        selected_factor_indices: np.ndarray
    ):
        """
        Args:
            beta: (n_assets, factor_dim) - factor loadings
            factor_cov: (factor_dim, factor_dim) - factor covariance matrix
            selected_factor_indices: indices of factors to use
        """
        self.beta_full = beta
        self.factor_cov_full = factor_cov
        self.selected_indices = selected_factor_indices

        # Filter to selected factors only
        self.beta = beta[:, selected_factor_indices]
        self.factor_cov = factor_cov[np.ix_(selected_factor_indices, selected_factor_indices)]

        self.n_assets = beta.shape[0]

    def compute_portfolio_risk(self, weights: np.ndarray) -> float:
        """
        Compute portfolio variance using factor model:
        Var(portfolio) = w^T @ beta @ Σ_factor @ beta^T @ w
        """
        portfolio_factor_exposure = weights @ self.beta  # (n_factors,)
        portfolio_variance = portfolio_factor_exposure @ self.factor_cov @ portfolio_factor_exposure
        return portfolio_variance

    def optimize(
        self,
        expected_returns: np.ndarray,
        max_position: float = 0.3,
        min_position: float = 0.0,
        target_return: float = None
    ) -> np.ndarray:
        """
        Mean-variance optimization

        Args:
            expected_returns: (n_assets,) expected returns
            max_position: maximum weight per asset
            min_position: minimum weight per asset
            target_return: target portfolio return (optional)

        Returns:
            weights: (n_assets,) optimal portfolio weights
        """
        print("\n=== Portfolio Optimization ===")
        print(f"Number of assets: {self.n_assets}")
        print(f"Selected factors: {len(self.selected_indices)}/{self.beta_full.shape[1]}")

        # Objective: minimize variance
        def objective(w):
            return self.compute_portfolio_risk(w)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # sum to 1
        ]

        # Target return constraint (optional)
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns - target_return
            })

        # Bounds
        bounds = [(min_position, max_position) for _ in range(self.n_assets)]

        # Initial guess: equal weight
        w0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        weights = result.x

        # Stats
        portfolio_return = weights @ expected_returns
        portfolio_risk = np.sqrt(self.compute_portfolio_risk(weights))
        sharpe = portfolio_return / (portfolio_risk + 1e-8)

        print(f"\nOptimized Portfolio:")
        print(f"  Expected Return: {portfolio_return:.4f}")
        print(f"  Volatility: {portfolio_risk:.4f}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Active positions: {(weights > 0.01).sum()}/{self.n_assets}")

        return weights


def run_factor_portfolio_pipeline(
    model_path: str,
    config_path: str,
    data_split: str = 'test',
    output_dir: str = 'results/factor_analysis'
):
    """
    Complete pipeline: factor extraction → analysis → portfolio optimization

    Args:
        model_path: path to trained embedding model
        config_path: path to config file
        data_split: 'train', 'val', or 'test'
        output_dir: directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    from models.embedding_model import FactorEmbeddingModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FactorEmbeddingModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {model_path}")

    # Load data
    from utils.data_loader import MarketDataLoader
    data_loader = MarketDataLoader(config)
    data_splits = data_loader.load_all_data()

    X_etf, X_macro, y = data_splits[data_split]

    # Extract factors
    print(f"\n=== Extracting Factors from {data_split} set ===")
    with torch.no_grad():
        X_etf_tensor = torch.FloatTensor(X_etf).to(device)
        X_macro_tensor = torch.FloatTensor(X_macro).to(device)
        factors = model(X_etf_tensor, X_macro_tensor).cpu().numpy()

    print(f"Factors shape: {factors.shape}")
    print(f"Returns shape: {y.shape}")

    # Analyze factors
    analyzer = FactorAnalyzer(factors, y)
    analyzer.estimate_factor_loadings()
    factor_metrics = analyzer.compute_factor_metrics()

    # Visualize
    analyzer.visualize_factors(save_path=f"{output_dir}/factor_analysis.png")

    # Select good factors
    selected_indices = analyzer.select_factors(
        min_return=0.0,
        min_sharpe=0.1,
        top_k_pct=0.7
    )

    # Save factor selection results
    factor_df = pd.DataFrame({
        'factor_id': range(analyzer.factor_dim),
        'return': analyzer.factor_returns,
        'volatility': analyzer.factor_volatility,
        'sharpe': analyzer.factor_sharpe,
        'selected': np.isin(range(analyzer.factor_dim), selected_indices)
    })
    factor_df.to_csv(f"{output_dir}/factor_metrics.csv", index=False)
    print(f"\nFactor metrics saved to {output_dir}/factor_metrics.csv")

    # Compute factor covariance
    factor_cov = np.cov(factors.T)

    # Portfolio optimization
    optimizer = PortfolioOptimizer(
        beta=analyzer.beta,
        factor_cov=factor_cov,
        selected_factor_indices=selected_indices
    )

    # Expected returns: use most recent factor values
    current_factors = factors[-1]
    expected_returns = analyzer.beta[:, selected_indices] @ current_factors[selected_indices]

    # Optimize
    weights = optimizer.optimize(
        expected_returns=expected_returns,
        max_position=0.25,
        min_position=0.0
    )

    # Save portfolio
    asset_names = config['data']['sector_etfs'] + config['data'].get('additional_etfs', [])
    portfolio_df = pd.DataFrame({
        'asset': asset_names,
        'weight': weights,
        'expected_return': expected_returns
    }).sort_values('weight', ascending=False)

    portfolio_df.to_csv(f"{output_dir}/optimal_portfolio.csv", index=False)
    print(f"\nPortfolio saved to {output_dir}/optimal_portfolio.csv")
    print("\n=== Top 10 Positions ===")
    print(portfolio_df.head(10).to_string(index=False))

    return {
        'factors': factors,
        'factor_metrics': factor_metrics,
        'selected_indices': selected_indices,
        'portfolio_weights': weights
    }


if __name__ == "__main__":
    # Example usage
    run_factor_portfolio_pipeline(
        model_path='results/checkpoints/best_model.pt',
        config_path='configs/config.yaml',
        data_split='test',
        output_dir='results/factor_analysis'
    )
