"""
lstm_predictor.py - ASTRA LSTM Trajectory Prediction Module
Bidirectional LSTM with attention mechanism for predicting multi-step
debris trajectories from historical orbital state sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over temporal LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        encoder_outputs: (B, T, H*2)  [bidirectional]
        Returns: context (B, H*2), weights (B, T)
        """
        scores  = self.v(torch.tanh(self.W(encoder_outputs))).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
        return context, weights


class DebrisLSTMPredictor(nn.Module):
    """
    Bidirectional LSTM trajectory predictor with:
      - Temporal attention mechanism
      - Residual skip connections
      - Multi-step prediction head
      - Uncertainty estimation (Monte Carlo Dropout)

    Input features per timestep:
      [x, y, z, vx, vy, vz, alt, speed, class_onehot×4]  → 14-dim
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 256,
                 num_layers: int = 3, output_dim: int = 6,
                 predict_steps: int = 30, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.predict_steps = predict_steps
        self.output_dim    = output_dim

        # Feature embedding
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional=True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # Attention
        self.attention = TemporalAttention(hidden_dim)

        # Decoder LSTM (unidirectional)
        self.decoder = nn.LSTM(
            input_size  = hidden_dim * 2 + output_dim,
            hidden_size = hidden_dim * 2,
            num_layers  = 1,
            batch_first = True,
        )

        # Output head: mean + log-variance (aleatoric uncertainty)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * 2),   # mu + log_sigma
        )

        # Residual projection
        self.res_proj = nn.Linear(input_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor,
                teacher_forcing: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T_in, input_dim)  – historical sequence
        teacher_forcing: (B, T_out, output_dim) – ground truth for training

        Returns:
          predictions: (B, T_out, output_dim) – predicted trajectory
          log_sigma:   (B, T_out, output_dim) – log uncertainty
          attn_weights:(B, T_in)              – attention weights
        """
        B, T_in, _ = x.shape

        # Encode
        feats = self.feature_proj(x)                     # (B,T,H)
        enc_out, (h_n, c_n) = self.encoder(feats)        # (B,T,2H)
        context, attn_w = self.attention(enc_out)         # (B,2H)

        # Init decoder state
        h_dec = context.unsqueeze(0).contiguous()
        c_dec = torch.zeros_like(h_dec)

        # Residual from last input
        last_x = x[:, -1, :]
        last_pred = self.res_proj(last_x)

        preds, log_sigmas = [], []
        for t in range(self.predict_steps):
            if teacher_forcing is not None and t > 0:
                dec_in = teacher_forcing[:, t-1, :]
            else:
                dec_in = last_pred
            dec_input  = torch.cat([context, dec_in], dim=-1).unsqueeze(1)
            out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            output     = self.output_head(out.squeeze(1))
            mu         = output[:, :self.output_dim]
            log_sigma  = output[:, self.output_dim:]
            last_pred  = mu
            preds.append(mu)
            log_sigmas.append(log_sigma)

        predictions = torch.stack(preds,      dim=1)   # (B, T_out, D)
        log_sigmas  = torch.stack(log_sigmas, dim=1)
        return predictions, log_sigmas, attn_w

    def predict_with_uncertainty(self, x: torch.Tensor,
                                  n_samples: int = 50) -> dict:
        """Monte Carlo Dropout uncertainty estimation."""
        self.train()  # enable dropout
        all_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds, log_sigma, _ = self.forward(x)
                all_preds.append(preds)
        self.eval()
        stack  = torch.stack(all_preds, dim=0)   # (S, B, T, D)
        mean   = stack.mean(dim=0)
        std    = stack.std(dim=0)
        return {"mean": mean, "std": std, "samples": stack}

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Dataset for supervised trajectory prediction training.
    Generates synthetic orbital trajectory sequences.
    """

    def __init__(self, n_samples=10000, seq_len=30, pred_len=30,
                 input_dim=14, noise_sigma=0.01):
        self.n_samples  = n_samples
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.input_dim  = input_dim
        self.noise      = noise_sigma
        self.rng        = np.random.default_rng(42)
        self._generate()

    def _generate(self):
        """Generate synthetic orbital sequences with perturbations."""
        total = self.seq_len + self.pred_len
        EARTH_MU = 3.986004418e14
        EARTH_R  = 6.371e6

        self.X, self.Y = [], []
        for _ in range(self.n_samples):
            alt  = self.rng.uniform(200e3, 2000e3)
            R    = EARTH_R + alt
            v    = np.sqrt(EARTH_MU / R)
            inc  = np.radians(self.rng.uniform(0, 98))
            RAAN = self.rng.uniform(0, 2*np.pi)
            nu0  = self.rng.uniform(0, 2*np.pi)
            omega= np.sqrt(EARTH_MU / R**3)

            seq = []
            for t in range(total):
                nu   = nu0 + omega * t
                # Simple circular orbit in orbital plane
                x_orb = R * np.cos(nu)
                y_orb = R * np.sin(nu)
                z_orb = 0.0
                # ECI rotation (simplified)
                cO,sO = np.cos(RAAN), np.sin(RAAN)
                ci,si = np.cos(inc),  np.sin(inc)
                pos = np.array([
                    cO*x_orb - sO*y_orb*ci,
                    sO*x_orb + cO*y_orb*ci,
                    y_orb*si,
                ])
                vel = np.array([
                    -cO*v*np.sin(nu) - sO*v*np.cos(nu)*ci,
                    -sO*v*np.sin(nu) + cO*v*np.cos(nu)*ci,
                     v*np.cos(nu)*si,
                ])
                # Feature vector: pos_norm, vel_norm, alt, speed, class_onehot
                pos_n = pos / (EARTH_R + 2000e3)
                vel_n = vel / 8000.0
                alt_n = (np.linalg.norm(pos)-EARTH_R) / 2000e3
                spd_n = np.linalg.norm(vel) / 8000.0
                cls   = np.zeros(4); cls[self.rng.integers(0,4)] = 1
                feat  = np.concatenate([pos_n, vel_n, [alt_n, spd_n], cls])
                feat += self.rng.normal(0, self.noise, feat.shape)
                seq.append(feat)

            seq = np.array(seq, dtype=np.float32)
            # target: [pos_norm, vel_norm] for prediction window
            target = seq[self.seq_len:, :6]
            self.X.append(seq[:self.seq_len])
            self.Y.append(target)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.Y[idx]))
