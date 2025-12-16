"""
Factor Embedding Model (Stage 1)
Pure embedding model: ETF + Macro → Factor Vector
No prediction head - just factor extraction
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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ETFEncoder(nn.Module):
    """Transformer encoder for all ETF returns (sector + additional)"""

    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config['model']['sector_encoder']['d_model']
        nhead = config['model']['sector_encoder']['nhead']
        num_layers = config['model']['sector_encoder']['num_layers']
        dim_feedforward = config['model']['sector_encoder']['dim_feedforward']
        dropout = config['model']['sector_encoder']['dropout']

        # Calculate total number of ETFs
        n_sector_etfs = len(config['data']['sector_etfs'])
        n_additional_etfs = len(config['data'].get('additional_etfs', []))
        n_total_etfs = n_sector_etfs + n_additional_etfs

        # Input projection
        self.input_proj = nn.Linear(n_total_etfs, self.d_model)

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
            x: (batch_size, seq_len, n_total_etfs)
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
    """Cross-attention layer to fuse ETF and macro information"""

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

    def forward(self, etf_enc, macro_enc):
        """
        Args:
            etf_enc: (seq_len, batch_size, d_model)
            macro_enc: (seq_len, batch_size, d_model)
        Returns:
            (seq_len, batch_size, d_model)
        """
        x = etf_enc

        for i, (attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
            # Cross-attention: ETF attends to macro
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


class FactorEmbeddingModel(nn.Module):
    """
    Factor Embedding Model (Stage 1 only)

    Input: ETF returns (25) + Macro indicators (10)
    Output: Factor embedding vector (128-dim)

    No prediction head - pure embedding extraction
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Encoders
        self.etf_encoder = ETFEncoder(config)
        self.macro_encoder = MacroEncoder(config)

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(config)

        # Factor extraction
        d_model = config['model']['cross_attention']['d_model']
        factor_dim = config['model']['factor_dim']
        self.factor_extractor = FactorExtractor(d_model, factor_dim)

    def forward(self, etf_data, macro_data):
        """
        Args:
            etf_data: (batch_size, seq_len, n_total_etfs) - all 25 ETFs
            macro_data: (batch_size, seq_len, n_macros)

        Returns:
            factors: (batch_size, factor_dim) - factor embedding
        """
        # Encode ETF and macro data
        etf_enc = self.etf_encoder(etf_data)  # (seq, batch, d_model)
        macro_enc = self.macro_encoder(macro_data)  # (seq, batch, d_model)

        # Cross-attention fusion
        fused = self.cross_attention(etf_enc, macro_enc)  # (seq, batch, d_model)

        # Extract factors
        factors = self.factor_extractor(fused)  # (batch, factor_dim)

        return factors

    def extract_factors_batch(self, etf_data, macro_data):
        """
        Convenience method for batch factor extraction
        """
        with torch.no_grad():
            return self.forward(etf_data, macro_data)


if __name__ == "__main__":
    # Test embedding model
    import yaml

    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = FactorEmbeddingModel(config)

    # Dummy input
    batch_size = 8
    seq_len = config['data']['lookback_window']
    n_sector = len(config['data']['sector_etfs'])
    n_additional = len(config['data']['additional_etfs'])
    n_total_etfs = n_sector + n_additional
    n_macros = len(config['data']['macro_indicators'])

    etf_data = torch.randn(batch_size, seq_len, n_total_etfs)
    macro_data = torch.randn(batch_size, seq_len, n_macros)

    # Forward pass
    factors = model(etf_data, macro_data)

    print(f"Input ETF shape: {etf_data.shape}")
    print(f"Input Macro shape: {macro_data.shape}")
    print(f"Output Factor shape: {factors.shape}")
    print(f"Factor dimension: {config['model']['factor_dim']}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
