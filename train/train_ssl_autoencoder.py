"""
Train a mask-aware autoencoder for self-supervised learning of protein embeddings.

python -m train.train_ssl_autoencoder \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint_dir checkpoints/ssl_autoencoder \
  --normalization_mode protein \
  --mask_prob 0.3 \
  --hidden_dim 256 \
  --latent_dim 64 \
  --dropout 0.1 \
  --batch_size 512 \
  --epochs 500 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --patience 50
"""

import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.ssl_autoencoder import MaskAwareAutoencoder, masked_mse_loss


class ProteinProfileDataset(Dataset):
    """
    Protein profile dataset for self-supervised masked reconstruction.

    Training:
        random mask is sampled each time.

    Validation:
        fixed random mask is generated once, so validation is deterministic.

    feature:
        concat(corrupted_filled_abundance, detection_mask)

    target:
        original standardized abundance with NaNs filled by 0

    ssl_mask:
        1 only at artificially masked observed entries
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mask_prob: float = 0.15,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
        normalization_mode: str = "protein",
        training: bool = True,
        seed: int = 42,
    ):
        self.proteins = df.index.tolist()
        self.X_raw = df.values.astype(np.float32)
        self.obs_mask = (~np.isnan(self.X_raw)).astype(np.float32)
        self.mask_prob = mask_prob
        self.training = training
        self.seed = seed

        self.normalization_mode = normalization_mode

        if normalization_mode == "patient":
            if means is None:
                means = np.nanmean(self.X_raw, axis=0)
            if stds is None:
                stds = np.nanstd(self.X_raw, axis=0)

            stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds)
            self.means = means.astype(np.float32)
            self.stds = stds.astype(np.float32)
            X_std = (self.X_raw - self.means) / self.stds
        elif normalization_mode == "protein":
            row_means = np.nanmean(self.X_raw, axis=1, keepdims=True)
            row_stds = np.nanstd(self.X_raw, axis=1, keepdims=True)
            row_stds = np.where((row_stds == 0) | np.isnan(row_stds), 1.0, row_stds)
            self.means = row_means.astype(np.float32)
            self.stds = row_stds.astype(np.float32)
            X_std = (self.X_raw - self.means) / self.stds
        elif normalization_mode == "none":
            self.means = None
            self.stds = None
            X_std = self.X_raw.copy()
        else:
            raise ValueError(
                f"Unknown normalization_mode={normalization_mode}. "
                "Choose from: protein, patient, none."
            )

        self.X_filled = np.where(np.isnan(X_std), 0.0, X_std).astype(np.float32)

        if not self.training:
            self.fixed_ssl_masks = self._build_fixed_val_masks(seed)
        else:
            self.fixed_ssl_masks = None

    def __len__(self):
        return self.X_filled.shape[0]

    def _sample_ssl_mask(self, obs: np.ndarray, rng=None) -> np.ndarray:
        observed_positions = np.where(obs > 0)[0]
        ssl_mask = np.zeros_like(obs, dtype=np.float32)

        if len(observed_positions) == 0:
            return ssl_mask

        n_mask = max(1, int(round(len(observed_positions) * self.mask_prob)))
        n_mask = min(n_mask, len(observed_positions))

        if rng is None:
            chosen = np.random.choice(observed_positions, size=n_mask, replace=False)
        else:
            chosen = rng.choice(observed_positions, size=n_mask, replace=False)

        ssl_mask[chosen] = 1.0
        return ssl_mask

    def _build_fixed_val_masks(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        masks = np.zeros_like(self.obs_mask, dtype=np.float32)

        for i in range(len(self.obs_mask)):
            masks[i] = self._sample_ssl_mask(self.obs_mask[i], rng=rng)

        return masks

    def __getitem__(self, idx):
        x = self.X_filled[idx].copy()
        obs = self.obs_mask[idx].copy()
        target = self.X_filled[idx].astype(np.float32)

        if self.training:
            ssl_mask = self._sample_ssl_mask(obs)
        else:
            ssl_mask = self.fixed_ssl_masks[idx].copy()

        # Important: hide the target values selected for SSL reconstruction.
        x[ssl_mask > 0] = 0.0

        feature = np.concatenate([x, obs], axis=0).astype(np.float32)

        return (
            torch.from_numpy(feature),
            torch.from_numpy(target),
            torch.from_numpy(ssl_mask),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_proteomics(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0)
        if df.shape[1] <= 1:
            df = pd.read_csv(path, index_col=0, sep="\t")
    except Exception:
        df = pd.read_csv(path, index_col=0, sep="\t")
    return df


def compute_patient_normalization(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_raw = train_df.values.astype(np.float32)
    means = np.nanmean(train_raw, axis=0).astype(np.float32)
    stds = np.nanstd(train_raw, axis=0).astype(np.float32)
    stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds).astype(np.float32)
    return means, stds


def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{epochs} [train]", leave=False)
    for feature, target, ssl_mask in pbar:
        feature  = feature.to(device)
        target   = target.to(device)
        ssl_mask = ssl_mask.to(device)

        pred, _ = model(feature)
        loss    = masked_mse_loss(pred, target, ssl_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, epoch, epochs):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{epochs} [val]  ", leave=False)
    for feature, target, ssl_mask in pbar:
        feature  = feature.to(device)
        target   = target.to(device)
        ssl_mask = ssl_mask.to(device)

        pred, _ = model(feature)
        loss    = masked_mse_loss(pred, target, ssl_mask)

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


def main(args):
    set_seed(args.seed)

    # ── Setup ────────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("── Load ──")
    df = load_proteomics(args.input)
    print(f"  Loaded: {df.shape[0]} proteins × {df.shape[1]} patients")

    n_total = len(df)
    n_val = int(n_total * args.val_fraction)
    n_train = n_total - n_val

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(args.seed)
    ).tolist()

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    means = None
    stds = None
    means_path = checkpoint_dir / "ssl_autoencoder_patient_means.npy"
    stds_path = checkpoint_dir / "ssl_autoencoder_patient_stds.npy"

    if args.normalization_mode == "patient":
        means, stds = compute_patient_normalization(train_df)
        np.save(means_path, means)
        np.save(stds_path, stds)
    elif args.normalization_mode not in {"protein", "none"}:
        raise ValueError("normalization_mode must be one of: protein, patient, none")

    train_dataset = ProteinProfileDataset(
        train_df,
        mask_prob=args.mask_prob,
        means=means,
        stds=stds,
        normalization_mode=args.normalization_mode,
        training=True,
        seed=args.seed,
    )

    val_dataset = ProteinProfileDataset(
        val_df,
        mask_prob=args.mask_prob,
        means=means,
        stds=stds,
        normalization_mode=args.normalization_mode,
        training=False,
        seed=args.seed + 1,
    )

    pin = device.type == "cuda"

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # ── Model ────────────────────────────────────────────────────────────────
    model = MaskAwareAutoencoder(
        n_patients=df.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    epochs_no_improve = 0
    best_path       = checkpoint_dir / "ssl_autoencoder_best.pt"
    last_path       = checkpoint_dir / "ssl_autoencoder_last.pt"
    train_log       = []

    def save_checkpoint(path):
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_patients":       df.shape[1],
            "hidden_dim":       args.hidden_dim,
            "latent_dim":       args.latent_dim,
            "dropout":          args.dropout,
            "normalization_mode": args.normalization_mode,
            "patient_means":     torch.from_numpy(means) if means is not None else None,
            "patient_stds":      torch.from_numpy(stds) if stds is not None else None,
            "patient_means_path": str(means_path),
            "patient_stds_path":  str(stds_path),
            "seed":              args.seed,
            "val_fraction":      args.val_fraction,
            "train_indices":     train_idx,
            "val_indices":       val_idx,
        }, path)

    print("── Training ──")
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_loss   = evaluate(model, val_loader, device, epoch, args.epochs)
        scheduler.step()

        train_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        epoch_pbar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}",
                               best=f"{best_val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            epochs_no_improve = 0
            save_checkpoint(best_path)
        else:
            epochs_no_improve += 1

        if args.patience and epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    save_checkpoint(last_path)

    # ── Loss plot ────────────────────────────────────────────────────────────
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(checkpoint_dir / "ssl_autoencoder_train_log.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"],   label="val")
    ax.axvline(log_df.loc[log_df["val_loss"].idxmin(), "epoch"],
               color="grey", linestyle="--", linewidth=0.8, label="best val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE Loss")
    ax.set_title("SSL Autoencoder Training")
    ax.legend()
    fig.tight_layout()
    plot_path = checkpoint_dir / "ssl_autoencoder_loss.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    tqdm.write(f"  Loss plot saved to {plot_path}")

    config = vars(args)
    config.update({"n_proteins": df.shape[0], "n_patients": df.shape[1],
                   "best_val_loss": best_val_loss})
    with open(checkpoint_dir / "ssl_autoencoder_config.json", "w") as f:
        json.dump(config, f, indent=2)

    tqdm.write(f"  Best val loss:   {best_val_loss:.6f}")
    tqdm.write(f"  Best checkpoint: {best_path}")
    tqdm.write(f"  Last checkpoint: {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",          type=str,   default="data/processed/proteomics_data_processed.csv")
    parser.add_argument("--checkpoint_dir", type=str,   required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--mask_prob",      type=float, default=0.15)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--latent_dim",     type=int,   default=32)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--normalization_mode", type=str, default="protein",
                        choices=["protein", "patient", "none"],
                        help=(
                            "protein: z-score each protein across patients; "
                            "patient: z-score each patient across train proteins; "
                            "none: use input values as-is."
                        ))
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--val_fraction",   type=float, default=0.1)
    parser.add_argument("--num_workers",    type=int,   default=0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--patience",       type=int,   default=None,
                        help="Early stopping patience (epochs). None = disabled.")
    parser.add_argument("--cpu",            action="store_true")

    args = parser.parse_args()
    main(args)
