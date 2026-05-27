# src/model/transformer_ssl.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatientTokenTransformer(nn.Module):
    """
    Transformer self-supervised model for protein profiles.

    Each protein is treated as a sequence of patient-level tokens.

    For one protein:
        token_p = [abundance_value_at_patient_p, detection_mask_at_patient_p]

    The transformer attends across patients and produces:
        1. reconstructed abundance values over patients
        2. one latent protein embedding
    """

    def __init__(
        self,
        n_patients: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_patients = n_patients
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Each patient token contains:
        #   standardized abundance value
        #   detection mask
        self.token_projection = nn.Linear(2, d_model)

        # Learnable patient-position embedding.
        # This allows the model to distinguish Patient_1, Patient_2, etc.
        self.patient_embedding = nn.Parameter(
            torch.randn(1, n_patients, d_model) * 0.02
        )

        # CLS token used as global protein representation.
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

        # Decode abundance from the global protein embedding plus patient ID.
        # This makes the exported z a true bottleneck for reconstruction,
        # instead of decoding directly from per-patient token states.
        self.latent_to_decoder = nn.Linear(latent_dim, d_model)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, obs_mask, ssl_mask=None):
        """
        Parameters
        ----------
        x:
            Tensor of shape [batch, n_patients].
            Standardized abundance values.
            Missing values should already be filled with 0.

        obs_mask:
            Tensor of shape [batch, n_patients].
            1 if originally observed, 0 if naturally missing.

        Returns
        -------
        recon:
            Tensor of shape [batch, n_patients].
            Reconstructed standardized abundance values.

        z:
            Tensor of shape [batch, latent_dim].
            Protein embedding.
        """
        batch_size, n_patients = x.shape

        if n_patients != self.n_patients:
            raise ValueError(
                f"Expected {self.n_patients} patients, got {n_patients}."
            )

        token_input = torch.stack([x, obs_mask], dim=-1)
        h = self.token_projection(token_input)
        h = h + self.patient_embedding

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)

        h = self.encoder(h)

        cls_h = h[:, 0, :]
        z = self.to_latent(cls_h)

        decoder_h = self.latent_to_decoder(z).unsqueeze(1)
        decoder_h = decoder_h + self.patient_embedding
        recon = self.decoder(decoder_h).squeeze(-1)

        return recon, z


def masked_mse_loss(pred, target, ssl_mask):
    """
    MSE loss only on artificially masked observed entries.

    pred:
        [batch, n_patients]

    target:
        [batch, n_patients]

    ssl_mask:
        [batch, n_patients]
        1 where an originally observed value was artificially masked.
        0 elsewhere.

    Natural missing values are not included in this loss.
    """
    denom = ssl_mask.sum().clamp(min=1.0)
    return (((pred - target) ** 2) * ssl_mask).sum() / denom


def cosine_consistency_loss(z1, z2):
    """
    Optional consistency loss.

    The same protein under two different random masks should produce
    similar embeddings.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return 1.0 - (z1 * z2).sum(dim=1).mean()
