"""
Transformer-based Factor Extraction Model for Asset Allocation
Architecture: Sector Encoder + Macro Encoder + Cross Attention + Factor Extraction
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SectorEncoder(nn.Module):
    """Transformer encoder for sector ETF returns"""

    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config['model']['sector_encoder']['d_model']
        nhead = config['model']['sector_encoder']['nhead']
        num_layers = config['model']['sector_encoder']['num_layers']
        dim_feedforward = config['model']['sector_encoder']['dim_feedforward']
        dropout = config['model']['sector_encoder']['dropout']

        # Input projection
        self.input_proj = nn.Linear(
            len(config['data']['sector_etfs']),
            self.d_model
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, n_sectors)
        Returns:
            (seq_len, batch_size, d_model)
        """
        # x: (batch, seq, features) -> (seq, batch, features)
        x = x.transpose(0, 1)

        # Project to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer_encoder(x)

        return x


class MacroEncoder(nn.Module):
    """Transformer encoder for macro indicators"""

    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config['model']['macro_encoder']['d_model']
        nhead = config['model']['macro_encoder']['nhead']
        num_layers = config['model']['macro_encoder']['num_layers']
        dim_feedforward = config['model']['macro_encoder']['dim_feedforward']
        dropout = config['model']['macro_encoder']['dropout']

        # Input projection
        self.input_proj = nn.Linear(
            len(config['data']['macro_indicators']),
            self.d_model
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, n_macros)
        Returns:
            (seq_len, batch_size, d_model)
        """
        # x: (batch, seq, features) -> (seq, batch, features)
        x = x.transpose(0, 1)

        # Project to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer_encoder(x)

        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention layer to fuse sector and macro information"""

    def __init__(self, config: dict):
        super().__init__()
        d_model = config['model']['cross_attention']['d_model']
        nhead = config['model']['cross_attention']['nhead']
        num_layers = config['model']['cross_attention']['num_layers']
        dropout = config['model']['cross_attention']['dropout']

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=False
            )
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers * 2)
        ])

    def forward(self, sector_enc, macro_enc):
        """
        Args:
            sector_enc: (seq_len, batch_size, d_model)
            macro_enc: (seq_len, batch_size, d_model)
        Returns:
            (seq_len, batch_size, d_model)
        """
        x = sector_enc

        for i, (attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
            # Cross-attention: sector attends to macro
            attn_out, _ = attn(
                query=x,
                key=macro_enc,
                value=macro_enc
            )
            x = self.norm_layers[i * 2](x + attn_out)

            # Feed-forward
            ffn_out = ffn(x)
            x = self.norm_layers[i * 2 + 1](x + ffn_out)

        return x


class FactorExtractor(nn.Module):
    """Extract factor vector via mean pooling and projection"""

    def __init__(self, d_model: int, factor_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, factor_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            factor: (batch_size, factor_dim)
        """
        # Mean pooling over sequence dimension
        x = x.mean(dim=0)  # (batch_size, d_model)

        # Project to factor dimension
        factor = self.projection(x)  # (batch_size, factor_dim)

        return factor


class PredictionHead(nn.Module):
    """Linear prediction head: r_hat = Θ * f_t"""

    def __init__(self, factor_dim: int, n_assets: int, hidden_dims: list = None):
        super().__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            # Simple linear prediction
            self.predictor = nn.Linear(factor_dim, n_assets)
        else:
            # MLP prediction
            layers = []
            in_dim = factor_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, n_assets))
            self.predictor = nn.Sequential(*layers)

    def forward(self, factor):
        """
        Args:
            factor: (batch_size, factor_dim)
        Returns:
            predictions: (batch_size, n_assets)
        """
        return self.predictor(factor)


class TransformerFactorModel(nn.Module):
    """
    Full model: Sector Encoder + Macro Encoder + Cross Attention + Factor Extraction + Prediction
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Encoders
        self.sector_encoder = SectorEncoder(config)
        self.macro_encoder = MacroEncoder(config)

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(config)

        # Factor extraction
        d_model = config['model']['cross_attention']['d_model']
        factor_dim = config['model']['factor_dim']
        self.factor_extractor = FactorExtractor(d_model, factor_dim)

        # Prediction head
        n_assets = len(config['data']['sector_etfs'])
        hidden_dims = config['model']['prediction'].get('hidden_dims', [])
        self.prediction_head = PredictionHead(factor_dim, n_assets, hidden_dims)

    def forward(self, sector_data, macro_data):
        """
        Args:
            sector_data: (batch_size, seq_len, n_sectors)
            macro_data: (batch_size, seq_len, n_macros)
        Returns:
            predictions: (batch_size, n_assets)
            factors: (batch_size, factor_dim)
        """
        # Encode sector and macro data
        sector_enc = self.sector_encoder(sector_data)  # (seq, batch, d_model)
        macro_enc = self.macro_encoder(macro_data)      # (seq, batch, d_model)

        # Cross-attention fusion
        fused = self.cross_attention(sector_enc, macro_enc)  # (seq, batch, d_model)

        # Extract factors
        factors = self.factor_extractor(fused)  # (batch, factor_dim)

        # Predict returns
        predictions = self.prediction_head(factors)  # (batch, n_assets)

        return predictions, factors

    def get_factor_weights(self):
        """Get the linear weights Θ for interpretability"""
        if isinstance(self.prediction_head.predictor, nn.Linear):
            return self.prediction_head.predictor.weight.data
        else:
            # Return the last layer weights for MLP
            return self.prediction_head.predictor[-1].weight.data


if __name__ == "__main__":
    # Test model
    import yaml

    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = TransformerFactorModel(config)

    # Dummy input
    batch_size = 8
    seq_len = config['data']['lookback_window']
    n_sectors = len(config['data']['sector_etfs'])
    n_macros = len(config['data']['macro_indicators'])

    sector_data = torch.randn(batch_size, seq_len, n_sectors)
    macro_data = torch.randn(batch_size, seq_len, n_macros)

    predictions, factors = model(sector_data, macro_data)

    print(f"Model created successfully!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Factors shape: {factors.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
