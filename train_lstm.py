"""
transformer_predictor.py - ASTRA Temporal Transformer Trajectory Predictor
Full encoder-decoder Transformer with causal masking and positional encoding
for long-horizon orbital debris trajectory forecasting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class OrbitalEmbedding(nn.Module):
    """
    Physics-informed orbital feature embedding.
    Encodes position, velocity, and orbital elements into transformer tokens.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Linear(3, d_model // 4), nn.GELU())
        self.vel_embed = nn.Sequential(
            nn.Linear(3, d_model // 4), nn.GELU())
        self.scalar_embed = nn.Sequential(
            nn.Linear(input_dim - 6, d_model // 4), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos_embed(x[..., :3])
        vel = self.vel_embed(x[..., 3:6])
        sca = self.scalar_embed(x[..., 6:])
        return self.fuse(torch.cat([pos, vel, sca], dim=-1))


class DebrisTransformerPredictor(nn.Module):
    """
    Full Encoder-Decoder Transformer for orbital trajectory prediction.

    Architecture:
      - Physics-informed embedding
      - Sinusoidal positional encoding
      - Multi-head self-attention encoder
      - Causal (masked) cross-attention decoder
      - Uncertainty quantification head

    Input:  historical state sequence  (B, T_in, F)
    Output: predicted trajectory       (B, T_out, 6)  [pos+vel normalised]
    """

    def __init__(self, input_dim: int = 14, d_model: int = 256,
                 n_heads: int = 8, n_encoder_layers: int = 6,
                 n_decoder_layers: int = 4, dim_feedforward: int = 1024,
                 predict_steps: int = 30, dropout: float = 0.1,
                 output_dim: int = 6):
        super().__init__()
        self.d_model       = d_model
        self.predict_steps = predict_steps
        self.output_dim    = output_dim

        # Embeddings
        self.src_embed  = OrbitalEmbedding(input_dim, d_model)
        self.tgt_embed  = nn.Linear(output_dim, d_model)
        self.src_pe     = PositionalEncoding(d_model, dropout=dropout)
        self.tgt_pe     = PositionalEncoding(d_model, dropout=dropout)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, n_encoder_layers,
            norm=nn.LayerNorm(d_model))
        self.decoder = nn.TransformerDecoder(
            dec_layer, n_decoder_layers,
            norm=nn.LayerNorm(d_model))

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim * 2),  # mu + log_sigma
        )

        # Learnable query tokens for prediction steps
        self.query_tokens = nn.Parameter(torch.randn(1, predict_steps, d_model))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src: torch.Tensor,
                tgt: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: (B, T_in, input_dim)
        tgt: (B, T_out, output_dim) [optional, for teacher forcing]
        Returns:
          mu:       (B, T_out, output_dim)
          log_sigma:(B, T_out, output_dim)
        """
        B = src.shape[0]

        # Encode source
        src_emb  = self.src_pe(self.src_embed(src))
        memory   = self.encoder(src_emb)

        # Decode with learnable query tokens
        queries  = self.query_tokens.expand(B, -1, -1)  # (B, T_out, D)
        if tgt is not None:
            tgt_emb  = self.tgt_pe(self.tgt_embed(tgt))
            tgt_mask = self._causal_mask(tgt.shape[1], src.device)
            dec_out  = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        else:
            dec_out  = self.decoder(queries, memory)

        out      = self.output_proj(dec_out)
        mu       = out[..., :self.output_dim]
        log_sigma= out[..., self.output_dim:]
        return mu, log_sigma

    def autoregressive_predict(self, src: torch.Tensor,
                                n_steps: Optional[int] = None) -> torch.Tensor:
        """Autoregressive inference without teacher forcing."""
        n = n_steps or self.predict_steps
        self.eval()
        with torch.no_grad():
            B = src.shape[0]
            src_emb = self.src_pe(self.src_embed(src))
            memory  = self.encoder(src_emb)

            # Seed decoder with zeros
            last_out = torch.zeros(B, 1, self.output_dim, device=src.device)
            preds    = []
            for t in range(n):
                tgt_emb = self.tgt_pe(self.tgt_embed(last_out))
                dec_out = self.decoder(tgt_emb, memory)
                out     = self.output_proj(dec_out[:, -1:])
                mu      = out[:, :, :self.output_dim]
                preds.append(mu)
                last_out = torch.cat([last_out, mu], dim=1)
            return torch.cat(preds, dim=1)   # (B, n, D)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HybridEnsemblePredictor:
    """
    Ensemble of LSTM + Transformer predictors with uncertainty-weighted fusion.
    Produces final trajectory prediction with combined epistemic/aleatoric uncertainty.
    """

    def __init__(self, lstm_model: nn.Module, transformer_model: nn.Module,
                 device: str = "cpu"):
        self.lstm        = lstm_model.to(device).eval()
        self.transformer = transformer_model.to(device).eval()
        self.device      = device
        self.weights     = {"lstm": 0.40, "transformer": 0.60}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        x: (B, T, F) input sequence
        Returns ensemble prediction dict with mean, std, and per-model outputs.
        """
        x = x.to(self.device)

        # LSTM prediction
        lstm_mu, lstm_log_sigma, _ = self.lstm(x)
        lstm_std = torch.exp(0.5 * lstm_log_sigma)

        # Transformer prediction
        transformer_mu, transformer_log_sigma = self.transformer(x)
        transformer_std = torch.exp(0.5 * transformer_log_sigma)

        # Uncertainty-weighted ensemble
        w_l = self.weights["lstm"]
        w_t = self.weights["transformer"]
        ensemble_mu  = w_l * lstm_mu  + w_t * transformer_mu
        ensemble_var = (w_l * (lstm_std**2 + lstm_mu**2) +
                        w_t * (transformer_std**2 + transformer_mu**2) -
                        ensemble_mu**2)
        ensemble_std = torch.sqrt(ensemble_var.clamp(min=0))

        return {
            "mean":             ensemble_mu.cpu().numpy(),
            "std":              ensemble_std.cpu().numpy(),
            "lstm_mean":        lstm_mu.cpu().numpy(),
            "transformer_mean": transformer_mu.cpu().numpy(),
        }
