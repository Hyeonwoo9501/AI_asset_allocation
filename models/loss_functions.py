"""
Composite Loss Functions for Factor Model
Includes: MSE (prediction), IC (ranking), Sharpe (portfolio), L1 (interpretability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error for prediction accuracy"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, n_assets)
            targets: (batch_size, n_assets)
        Returns:
            scalar loss
        """
        return F.mse_loss(predictions, targets)


class ICLoss(nn.Module):
    """
    Information Coefficient (IC) Loss - measures ranking correlation
    Negative Spearman correlation (we want to maximize IC, so minimize negative IC)
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, n_assets)
            targets: (batch_size, n_assets)
        Returns:
            scalar loss (negative IC)
        """
        # Compute rank correlation (Spearman-like)
        batch_size = predictions.shape[0]

        # Rank predictions and targets
        pred_ranks = self._get_ranks(predictions)
        target_ranks = self._get_ranks(targets)

        # Compute correlation
        pred_centered = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
        target_centered = target_ranks - target_ranks.mean(dim=1, keepdim=True)

        numerator = (pred_centered * target_centered).sum(dim=1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1))

        ic = numerator / (pred_std * target_std + 1e-8)
        ic = ic.mean()  # Average over batch

        # Return negative IC (we want to minimize loss)
        return -ic

    def _get_ranks(self, x):
        """Convert values to ranks"""
        # argsort twice gives ranks
        sorted_indices = torch.argsort(x, dim=1)
        ranks = torch.argsort(sorted_indices, dim=1).float()
        return ranks


class SharpeLoss(nn.Module):
    """
    Portfolio Sharpe Ratio Loss
    Constructs a long-short portfolio based on predictions and computes Sharpe
    """

    def __init__(self, top_k: int = 5, transaction_cost: float = 0.001):
        super().__init__()
        self.top_k = top_k
        self.transaction_cost = transaction_cost

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, n_assets)
            targets: (batch_size, n_assets) - actual returns
        Returns:
            scalar loss (negative Sharpe)
        """
        batch_size, n_assets = predictions.shape

        # Construct portfolio weights based on predictions
        # Long top-k, short bottom-k
        weights = self._get_portfolio_weights(predictions)  # (batch_size, n_assets)

        # Calculate portfolio returns
        portfolio_returns = (weights * targets).sum(dim=1)  # (batch_size,)

        # Subtract transaction costs (simplified)
        portfolio_returns = portfolio_returns - self.transaction_cost * weights.abs().sum(dim=1)

        # Compute Sharpe ratio
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-8

        sharpe = mean_return / std_return

        # Return negative Sharpe (we want to minimize loss)
        return -sharpe

    def _get_portfolio_weights(self, predictions):
        """
        Create portfolio weights: long top-k, short bottom-k
        Returns normalized weights
        """
        batch_size, n_assets = predictions.shape

        # Get top-k and bottom-k indices
        _, top_indices = torch.topk(predictions, self.top_k, dim=1)
        _, bottom_indices = torch.topk(predictions, self.top_k, dim=1, largest=False)

        # Initialize weights
        weights = torch.zeros_like(predictions)

        # Long top-k (equal weight)
        weights.scatter_(1, top_indices, 1.0 / self.top_k)

        # Short bottom-k (equal weight)
        weights.scatter_(1, bottom_indices, -1.0 / self.top_k)

        return weights


class L1RegularizationLoss(nn.Module):
    """L1 regularization on model parameters for interpretability"""

    def __init__(self):
        super().__init__()

    def forward(self, model):
        """
        Args:
            model: the transformer factor model
        Returns:
            scalar loss (L1 norm of prediction head weights)
        """
        # Apply L1 penalty on prediction head weights (Θ)
        weights = model.get_factor_weights()
        l1_loss = torch.abs(weights).sum()

        return l1_loss


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple objectives:
    L = λ1*MSE + λ2*(-IC) + λ3*(-Sharpe) + λ4*L1
    """

    def __init__(self, config: dict):
        super().__init__()

        self.mse_weight = config['loss']['mse_weight']
        self.ic_weight = config['loss']['ic_weight']
        self.sharpe_weight = config['loss']['sharpe_weight']
        self.l1_weight = config['loss']['l1_weight']

        self.mse_loss = MSELoss()
        self.ic_loss = ICLoss()
        self.sharpe_loss = SharpeLoss(
            top_k=config['portfolio']['top_k'],
            transaction_cost=config['portfolio']['transaction_cost']
        )
        self.l1_loss = L1RegularizationLoss()

    def forward(self, predictions, targets, model):
        """
        Args:
            predictions: (batch_size, n_assets)
            targets: (batch_size, n_assets)
            model: the transformer factor model
        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        # Compute individual losses
        mse = self.mse_loss(predictions, targets)
        ic = self.ic_loss(predictions, targets)
        sharpe = self.sharpe_loss(predictions, targets)
        l1 = self.l1_loss(model)

        # Weighted combination
        total_loss = (
            self.mse_weight * mse +
            self.ic_weight * ic +
            self.sharpe_weight * sharpe +
            self.l1_weight * l1
        )

        # Return total loss and individual components
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse.item(),
            'ic': -ic.item(),  # Report positive IC
            'sharpe': -sharpe.item(),  # Report positive Sharpe
            'l1': l1.item()
        }

        return total_loss, loss_dict


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics
    Args:
        predictions: (batch_size, n_assets) or (n_samples, n_assets)
        targets: (batch_size, n_assets) or (n_samples, n_assets)
    Returns:
        dict of metrics
    """
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()

    # MSE and MAE
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()

    # Information Coefficient (Spearman correlation)
    ic_list = []
    for i in range(predictions.shape[0]):
        pred_ranks = torch.argsort(torch.argsort(predictions[i]))
        target_ranks = torch.argsort(torch.argsort(targets[i]))

        # Pearson correlation on ranks
        pred_centered = pred_ranks.float() - pred_ranks.float().mean()
        target_centered = target_ranks.float() - target_ranks.float().mean()

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

        ic = (numerator / (denominator + 1e-8)).item()
        ic_list.append(ic)

    mean_ic = sum(ic_list) / len(ic_list)

    # Long-short portfolio return
    top_k = 3
    portfolio_returns = []
    for i in range(predictions.shape[0]):
        # Long top-k
        _, top_idx = torch.topk(predictions[i], top_k)
        # Short bottom-k
        _, bottom_idx = torch.topk(predictions[i], top_k, largest=False)

        port_ret = targets[i][top_idx].mean() - targets[i][bottom_idx].mean()
        portfolio_returns.append(port_ret.item())

    mean_port_return = sum(portfolio_returns) / len(portfolio_returns)

    return {
        'mse': mse,
        'mae': mae,
        'ic': mean_ic,
        'portfolio_return': mean_port_return
    }


if __name__ == "__main__":
    # Test loss functions
    batch_size = 32
    n_assets = 11

    predictions = torch.randn(batch_size, n_assets)
    targets = torch.randn(batch_size, n_assets)

    # Test individual losses
    mse_loss = MSELoss()
    ic_loss = ICLoss()
    sharpe_loss = SharpeLoss(top_k=3)

    print("MSE Loss:", mse_loss(predictions, targets).item())
    print("IC Loss:", ic_loss(predictions, targets).item())
    print("Sharpe Loss:", sharpe_loss(predictions, targets).item())

    # Test metrics
    metrics = compute_metrics(predictions, targets)
    print("\nMetrics:", metrics)
