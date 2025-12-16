"""
Factor Selection Module
Filters out high-volatility, low-return factors to improve portfolio stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorSelector(nn.Module):
    """
    Select factors based on return contribution and volatility
    Strategy: Keep high-return, low-volatility factors; filter out the rest
    """

    def __init__(self, config: dict):
        super().__init__()

        factor_dim = config['model']['factor_dim']
        self.factor_dim = factor_dim

        # Configuration
        factor_config = config['model']['factor_selection']
        self.enabled = factor_config['enabled']
        self.method = factor_config['method']
        self.min_sharpe = factor_config.get('min_sharpe', 0.3)
        self.volatility_penalty = factor_config.get('volatility_penalty', 1.0)

        # Learnable gating network
        self.gate_network = nn.Sequential(
            nn.Linear(factor_dim, factor_dim * 2),
            nn.LayerNorm(factor_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(factor_dim * 2, factor_dim),
            nn.Sigmoid()  # 0-1 gate values
        )

        # Factor quality estimator (predicts factor's expected return)
        self.quality_estimator = nn.Sequential(
            nn.Linear(factor_dim, factor_dim),
            nn.ReLU(),
            nn.Linear(factor_dim, factor_dim)
        )

        # Volatility estimator (predicts factor's volatility)
        self.volatility_estimator = nn.Sequential(
            nn.Linear(factor_dim, factor_dim),
            nn.ReLU(),
            nn.Linear(factor_dim, factor_dim),
            nn.Softplus()  # ensure positive volatility
        )

        # Running statistics for normalization
        self.register_buffer('running_mean', torch.zeros(factor_dim))
        self.register_buffer('running_var', torch.ones(factor_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0))

    def forward(self, factors, return_info=False):
        """
        Args:
            factors: (batch_size, factor_dim)
            return_info: whether to return selection information

        Returns:
            selected_factors: (batch_size, factor_dim) - weighted factors
            info: dict (optional) - selection weights and statistics
        """
        if not self.enabled:
            if return_info:
                return factors, {'selection_weights': torch.ones_like(factors)}
            return factors

        batch_size = factors.shape[0]

        # Update running statistics (during training)
        if self.training:
            self._update_running_stats(factors)

        # Normalize factors
        normalized_factors = (factors - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)

        # Compute selection weights based on method
        if self.method == 'adaptive':
            selection_weights = self._adaptive_selection(normalized_factors)
        elif self.method == 'sharpe':
            selection_weights = self._sharpe_selection(normalized_factors)
        elif self.method == 'return_vol_ratio':
            selection_weights = self._return_vol_selection(normalized_factors)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        # Apply selection weights
        selected_factors = factors * selection_weights

        if return_info:
            info = {
                'selection_weights': selection_weights,
                'raw_factors': factors,
                'selected_factors': selected_factors,
                'mean_weight': selection_weights.mean(dim=0),
                'num_active_factors': (selection_weights.mean(dim=0) > 0.1).sum().item()
            }
            return selected_factors, info

        return selected_factors

    def _adaptive_selection(self, factors):
        """
        Adaptive selection: learnable gates + volatility penalty

        Strategy:
        1. Gate network learns which factors are useful
        2. Volatility estimator penalizes high-volatility factors
        3. Quality estimator boosts high-quality factors

        weight = gate * quality / (volatility + eps)
        """
        # Learnable gates (which factors to use)
        gates = self.gate_network(factors)  # (batch, factor_dim)

        # Estimate factor quality (expected return contribution)
        quality = self.quality_estimator(factors)  # (batch, factor_dim)
        quality = torch.tanh(quality)  # normalize to [-1, 1]

        # Estimate factor volatility
        volatility = self.volatility_estimator(factors)  # (batch, factor_dim)

        # Compute selection weights
        # High quality + low volatility → high weight
        quality_score = F.relu(quality)  # only positive contributions
        volatility_penalty = 1.0 / (1.0 + self.volatility_penalty * volatility)

        weights = gates * quality_score * volatility_penalty

        # Normalize weights to sum to factor_dim (preserve scale)
        weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-8)

        return weights

    def _sharpe_selection(self, factors):
        """
        Sharpe-based selection: keep factors with high sharpe ratio

        sharpe = quality / (volatility + eps)
        weight = ReLU(sharpe - threshold)
        """
        # Estimate quality and volatility
        quality = self.quality_estimator(factors)
        quality = torch.tanh(quality)

        volatility = self.volatility_estimator(factors)

        # Compute sharpe ratio
        sharpe = quality / (volatility + 1e-8)

        # Threshold: keep only factors above min_sharpe
        weights = F.relu(sharpe - self.min_sharpe)

        # Normalize
        weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-8)

        return weights

    def _return_vol_selection(self, factors):
        """
        Return/Volatility ratio selection

        Select top-K factors by return/volatility ratio
        """
        # Estimate quality and volatility
        quality = self.quality_estimator(factors)
        quality = F.relu(torch.tanh(quality))  # positive only

        volatility = self.volatility_estimator(factors)

        # Compute ratio
        ratio = quality / (volatility + 1e-8)

        # Select top 70% factors
        k = int(self.factor_dim * 0.7)
        top_k_values, top_k_indices = torch.topk(ratio, k, dim=-1)

        # Create binary mask
        weights = torch.zeros_like(factors)
        weights.scatter_(-1, top_k_indices, 1.0)

        # Weight by inverse volatility
        inv_vol_weights = 1.0 / (volatility + 1e-8)
        weights = weights * inv_vol_weights

        # Normalize
        weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-8)

        return weights

    def _update_running_stats(self, factors):
        """Update running mean and variance for normalization"""
        with torch.no_grad():
            batch_mean = factors.mean(dim=0)
            batch_var = factors.var(dim=0, unbiased=False)

            # Exponential moving average
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var

            self.num_batches_tracked += 1


class FactorQualityLoss(nn.Module):
    """
    Additional loss to train factor selector
    Encourages:
    1. High-quality factors to have positive correlation with returns
    2. Low-volatility factors to be preferred
    """

    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config['model']['factor_selection']['enabled']
        self.volatility_penalty = config['model']['factor_selection'].get('volatility_penalty', 1.0)

    def forward(self, factor_info, returns):
        """
        Args:
            factor_info: dict from FactorSelector (with return_info=True)
            returns: (batch, n_assets) actual returns

        Returns:
            loss: scalar
        """
        if not self.enabled or factor_info is None:
            return torch.tensor(0.0)

        selection_weights = factor_info['selection_weights']  # (batch, factor_dim)

        # Encourage diversity: don't select only a few factors
        mean_weights = selection_weights.mean(dim=0)  # (factor_dim,)
        diversity_loss = -torch.log(mean_weights + 1e-8).mean()

        # Encourage sparsity: don't use all factors
        sparsity_loss = selection_weights.mean()

        # Balance: want ~70% of factors active
        target_activation = 0.7
        activation_rate = (mean_weights > 0.1).float().mean()
        balance_loss = (activation_rate - target_activation) ** 2

        total_loss = 0.1 * diversity_loss + 0.1 * sparsity_loss + 0.5 * balance_loss

        return total_loss


if __name__ == "__main__":
    # Test factor selector
    import yaml

    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create selector
    selector = FactorSelector(config)

    # Test forward pass
    batch_size = 32
    factor_dim = config['model']['factor_dim']

    factors = torch.randn(batch_size, factor_dim)

    # Without info
    selected = selector(factors)
    print(f"Input shape: {factors.shape}")
    print(f"Output shape: {selected.shape}")

    # With info
    selected, info = selector(factors, return_info=True)
    print(f"\nSelection info:")
    print(f"  Mean weight per factor: {info['mean_weight'].mean():.4f}")
    print(f"  Active factors: {info['num_active_factors']}/{factor_dim}")
    print(f"  Weight std: {info['mean_weight'].std():.4f}")
