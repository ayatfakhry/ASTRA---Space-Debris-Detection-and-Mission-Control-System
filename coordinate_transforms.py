"""
anomaly_detector.py - ASTRA Anomaly Detection via Variational Autoencoder
Detects anomalous debris behaviour (unexpected manoeuvres, fragmentation events,
sensor spoofing) using reconstruction error thresholding on orbital state sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class OrbitalEncoder(nn.Module):
    """Temporal convolutional encoder for orbital state sequences."""

    def __init__(self, input_dim: int, latent_dim: int, seq_len: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64,        128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128,       256, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(256)
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F) → (B, F, T) for Conv1d
        x = x.permute(0,2,1)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)   # (B, 256)
        return self.fc_mu(x), self.fc_logvar(x)


class OrbitalDecoder(nn.Module):
    """Temporal convolutional decoder for sequence reconstruction."""

    def __init__(self, latent_dim: int, output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc      = nn.Linear(latent_dim, 256)
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(128, 64,  kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose1d(64, output_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc(z))
        x = x.unsqueeze(-1).expand(-1,-1,self.seq_len)
        x = F.gelu(self.bn1(self.deconv1(x)))
        x = F.gelu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        return x.permute(0,2,1)   # (B, T, output_dim)


class OrbitalVAE(nn.Module):
    """
    Variational Autoencoder for orbital trajectory anomaly detection.
    Normal debris follows predictable Keplerian paths; anomalies
    (manoeuvres, fragmentation, data corruption) yield high reconstruction error.
    """

    def __init__(self, input_dim: int = 14, latent_dim: int = 32,
                 seq_len: int = 30):
        super().__init__()
        self.encoder = OrbitalEncoder(input_dim, latent_dim, seq_len)
        self.decoder = OrbitalDecoder(latent_dim, input_dim, seq_len)
        self.latent_dim = latent_dim
        self.input_dim  = input_dim
        self.seq_len    = seq_len

    def reparameterize(self, mu: torch.Tensor,
                        logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ."""
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        return mu

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decoder(z)
        return x_recon, mu, logvar

    def elbo_loss(self, x: torch.Tensor, beta: float = 1.0
                  ) -> Tuple[torch.Tensor, dict]:
        """β-VAE ELBO loss: reconstruction + β·KL divergence."""
        x_recon, mu, logvar = self.forward(x)
        recon = F.mse_loss(x_recon, x, reduction="mean")
        kl    = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()
        loss  = recon + beta*kl
        return loss, {"recon": recon.item(), "kl": kl.item()}

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """Per-sample reconstruction error (MSE over time and features)."""
        self.eval()
        x_recon, _, _ = self.forward(x)
        err = ((x - x_recon)**2).mean(dim=(1,2))  # (B,)
        return err.cpu().numpy()

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AnomalyDetector:
    """
    Real-time orbital anomaly detection engine.
    Maintains rolling reconstruction error statistics and
    flags objects exceeding adaptive threshold (μ + k·σ).
    """

    ANOMALY_TYPES = {
        "MANOEUVRE":     "Intentional velocity change detected",
        "FRAGMENTATION": "Sudden increase in object multiplicity",
        "SENSOR_FAULT":  "Measurement inconsistency with propagated state",
        "REENTRY":       "Rapid altitude decrease below threshold",
        "UNKNOWN":       "Unclassified anomalous behaviour",
    }

    def __init__(self, model: OrbitalVAE, threshold_sigma: float = 3.0,
                 device: str = "cpu"):
        self.model    = model.to(device).eval()
        self.device   = device
        self.k_sigma  = threshold_sigma
        self.error_history: List[float] = []
        self.anomaly_log:   List[dict]  = []
        self._threshold = 0.05   # initialised conservatively

    def update_threshold(self):
        """Adapt threshold from rolling error statistics."""
        if len(self.error_history) > 50:
            mu    = np.mean(self.error_history[-500:])
            sigma = np.std( self.error_history[-500:])
            self._threshold = mu + self.k_sigma*sigma

    @torch.no_grad()
    def detect(self, sequences: np.ndarray,
               object_ids: Optional[List[int]] = None,
               t: float = 0.0) -> List[dict]:
        """
        Detect anomalies in a batch of orbital state sequences.

        sequences: (N, T, F) numpy array of orbital states
        Returns list of anomaly records for flagged objects.
        """
        x      = torch.from_numpy(sequences.astype(np.float32)).to(self.device)
        errors = self.model.reconstruction_error(x)
        self.error_history.extend(errors.tolist())
        self.update_threshold()

        anomalies = []
        for i, err in enumerate(errors):
            obj_id = object_ids[i] if object_ids else i
            if err > self._threshold:
                a_type = self._classify_anomaly(sequences[i], err)
                record = {
                    "object_id":    obj_id,
                    "timestamp":    t,
                    "recon_error":  float(err),
                    "threshold":    float(self._threshold),
                    "severity":     float(err / (self._threshold+1e-9)),
                    "anomaly_type": a_type,
                    "description":  self.ANOMALY_TYPES[a_type],
                }
                self.anomaly_log.append(record)
                anomalies.append(record)
                logger.warning(f"ANOMALY [{a_type}] Object {obj_id} | "
                               f"Error={err:.4f} > Thresh={self._threshold:.4f}")
        return anomalies

    @staticmethod
    def _classify_anomaly(seq: np.ndarray, error: float) -> str:
        """Heuristic anomaly classification based on sequence statistics."""
        vel  = seq[:, 3:6]
        dv   = np.diff(vel, axis=0)
        alt  = seq[:, 6] if seq.shape[1] > 6 else np.zeros(seq.shape[0])
        if np.linalg.norm(dv).max() > 0.5:
            return "MANOEUVRE"
        if alt[-1] < 0.1 and alt[0] > 0.3:
            return "REENTRY"
        if error > 1.0:
            return "SENSOR_FAULT"
        return "UNKNOWN"

    def get_summary(self) -> dict:
        return {
            "total_anomalies":   len(self.anomaly_log),
            "current_threshold": round(self._threshold, 6),
            "recent_anomalies":  self.anomaly_log[-10:],
        }
