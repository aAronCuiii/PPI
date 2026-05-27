# src/models/ssl_autoencoder.py

import torch
import torch.nn as nn


class MaskAwareAutoencoder(nn.Module):
    """
    Mask-aware denoising autoencoder for protein abundance profiles.

    Input:
        feature = concat(filled_abundance, detection_mask)

    If there are 118 patients:
        filled_abundance: 118
        detection_mask:   118
        total input dim:  236

    Output:
        reconstruction of abundance profile over patients
        latent protein embedding
    """

    def __init__(
        self,
        n_patients: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_patients = n_patients
        input_dim = 2 * n_patients

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, n_patients),
        )

    def forward(self, feature):
        z = self.encoder(feature)
        recon = self.decoder(z)
        return recon, z


def masked_mse_loss(pred, target, ssl_mask):
    """
    MSE loss only on artificially masked observed entries.

    pred:
        reconstructed abundance, shape [batch, n_patients]

    target:
        true standardized abundance, shape [batch, n_patients]

    ssl_mask:
        1 for artificially masked observed values
        0 otherwise

    Natural NaNs are never included in the loss.
    """
    denom = ssl_mask.sum().clamp(min=1.0)
    loss = (((pred - target) ** 2) * ssl_mask).sum() / denom
    return loss