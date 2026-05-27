"""
Train a patient-token transformer for self-supervised protein embedding learning.

python -m train.train_ssl_transformer \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint_dir checkpoints/ssl_transformer \
  --normalization_mode protein \
  --mask_prob 0.3 \
  --d_model 64 \
  --n_heads 4 \
  --n_layers 2 \
  --dim_feedforward 128 \
  --latent_dim 64 \
  --dropout 0.1 \
  --batch_size 256 \
  --epochs 500 \
  --lr 5e-4 \
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

from src.model.ssl_transformer import (
    PatientTokenTransformer,
    masked_mse_loss,
    cosine_consistency_loss,
)


class ProteinTransformerDataset(Dataset):
    """
    Dataset for patient-token transformer SSL.

    For each protein:
        x: corrupted standardized abundance, natural NaNs filled by 0
        obs: original detection mask
        target: original standardized abundance, natural NaNs filled by 0
        ssl_mask: artificially masked observed entries

    Training:
        random mask is sampled every __getitem__

    Validation:
        fixed random mask is generated once in __init__
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
            self.fixed_ssl_masks = self._build_fixed_val_masks(seed=seed)
        else:
            self.fixed_ssl_masks = None

    def __len__(self):
        return self.X_filled.shape[0]

    def _sample_ssl_mask(self, obs: np.ndarray, rng=None) -> np.ndarray:
        """
        Sample a mask only from originally observed positions.
        """
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
        """
        Build deterministic validation masks once.
        """
        rng = np.random.default_rng(seed)
        masks = np.zeros_like(self.obs_mask, dtype=np.float32)

        for i in range(len(self.obs_mask)):
            masks[i] = self._sample_ssl_mask(self.obs_mask[i], rng=rng)

        return masks

    def _make_view(self, idx):
        x = self.X_filled[idx].copy()
        obs = self.obs_mask[idx].copy()
        target = self.X_filled[idx].astype(np.float32)

        if self.training:
            ssl_mask = self._sample_ssl_mask(obs)
        else:
            ssl_mask = self.fixed_ssl_masks[idx].copy()

        # Important: hide target values at SSL-masked positions.
        x[ssl_mask > 0] = 0.0

        return x, obs, target, ssl_mask

    def __getitem__(self, idx):
        x, obs, target, ssl_mask = self._make_view(idx)
        return (
            torch.from_numpy(x),
            torch.from_numpy(obs),
            torch.from_numpy(target),
            torch.from_numpy(ssl_mask),
        )

class ProteinTransformerTwoViewDataset(ProteinTransformerDataset):
    """
    Two-view dataset for consistency loss.

    Training:
        two independently masked random views.

    Validation:
        deterministic fixed views. For consistency loss, this is not very useful,
        so validation consistency should be interpreted carefully.
    """

    def __getitem__(self, idx):
        x1, obs1, target1, ssl_mask1 = self._make_view(idx)

        if self.training:
            x2, obs2, target2, ssl_mask2 = self._make_view(idx)
        else:
            # In validation, use the same fixed view twice.
            # This makes validation deterministic.
            x2, obs2, target2, ssl_mask2 = (
                x1.copy(),
                obs1.copy(),
                target1.copy(),
                ssl_mask1.copy(),
            )

        return (
            torch.from_numpy(x1), torch.from_numpy(obs1),
            torch.from_numpy(target1), torch.from_numpy(ssl_mask1),
            torch.from_numpy(x2), torch.from_numpy(obs2),
            torch.from_numpy(target2), torch.from_numpy(ssl_mask2),
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


def train_one_epoch_single_view(model, loader, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{epochs} [train]", leave=False)
    for x, obs, target, ssl_mask in pbar:
        x        = x.to(device)
        obs      = obs.to(device)
        target   = target.to(device)
        ssl_mask = ssl_mask.to(device)

        pred, _ = model(x, obs, ssl_mask)
        loss    = masked_mse_loss(pred, target, ssl_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


def train_one_epoch_two_view(model, loader, optimizer, device, epoch, epochs,
                              lambda_consistency: float = 0.1):
    model.train()
    total_loss        = 0.0
    total_recon       = 0.0
    total_consistency = 0.0
    n_batches         = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{epochs} [train]", leave=False)
    for batch in pbar:
        x1, obs1, target1, ssl_mask1, x2, obs2, target2, ssl_mask2 = batch

        x1, obs1, target1, ssl_mask1 = (
            x1.to(device), obs1.to(device), target1.to(device), ssl_mask1.to(device)
        )
        x2, obs2, target2, ssl_mask2 = (
            x2.to(device), obs2.to(device), target2.to(device), ssl_mask2.to(device)
        )

        pred1, z1   = model(x1, obs1, ssl_mask1)
        pred2, z2   = model(x2, obs2, ssl_mask2)
        recon_loss  = 0.5 * (masked_mse_loss(pred1, target1, ssl_mask1) +
                             masked_mse_loss(pred2, target2, ssl_mask2))
        cons_loss   = cosine_consistency_loss(z1, z2)
        loss        = recon_loss + lambda_consistency * cons_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss        += loss.item()
        total_recon       += recon_loss.item()
        total_consistency += cons_loss.item()
        n_batches         += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    n = max(n_batches, 1)
    return {"loss": total_loss / n, "recon_loss": total_recon / n,
            "consistency_loss": total_consistency / n}


@torch.no_grad()
def evaluate_single_view(model, loader, device, epoch, epochs):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{epochs} [val]  ", leave=False)
    for x, obs, target, ssl_mask in pbar:
        x, obs, target, ssl_mask = (
            x.to(device), obs.to(device), target.to(device), ssl_mask.to(device)
        )
        pred, _ = model(x, obs, ssl_mask)
        loss    = masked_mse_loss(pred, target, ssl_mask)
        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_two_view(model, loader, device, epoch, epochs,
                      lambda_consistency: float = 0.1):
    model.eval()
    total_loss        = 0.0
    total_recon       = 0.0
    total_consistency = 0.0
    n_batches         = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d}/{epochs} [val]  ", leave=False)
    for batch in pbar:
        x1, obs1, target1, ssl_mask1, x2, obs2, target2, ssl_mask2 = batch
        x1, obs1, target1, ssl_mask1 = (
            x1.to(device), obs1.to(device), target1.to(device), ssl_mask1.to(device)
        )
        x2, obs2, target2, ssl_mask2 = (
            x2.to(device), obs2.to(device), target2.to(device), ssl_mask2.to(device)
        )

        pred1, z1   = model(x1, obs1, ssl_mask1)
        pred2, z2   = model(x2, obs2, ssl_mask2)
        recon_loss  = 0.5 * (masked_mse_loss(pred1, target1, ssl_mask1) +
                             masked_mse_loss(pred2, target2, ssl_mask2))
        cons_loss   = cosine_consistency_loss(z1, z2)
        loss        = recon_loss + lambda_consistency * cons_loss

        total_loss        += loss.item()
        total_recon       += recon_loss.item()
        total_consistency += cons_loss.item()
        n_batches         += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    n = max(n_batches, 1)
    return {"loss": total_loss / n, "recon_loss": total_recon / n,
            "consistency_loss": total_consistency / n}


def plot_loss(log_df: pd.DataFrame, use_consistency: bool, save_path: Path):
    if use_consistency:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, col, title in zip(
            axes,
            ["loss", "recon_loss", "consistency_loss"],
            ["Total Loss", "Reconstruction Loss", "Consistency Loss"],
        ):
            ax.plot(log_df["epoch"], log_df[f"train_{col}"], label="train")
            ax.plot(log_df["epoch"], log_df[f"val_{col}"],   label="val")
            ax.axvline(log_df.loc[log_df[f"val_{col}"].idxmin(), "epoch"],
                       color="grey", linestyle="--", linewidth=0.8, label="best val")
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
        ax.plot(log_df["epoch"], log_df["val_loss"],   label="val")
        ax.axvline(log_df.loc[log_df["val_loss"].idxmin(), "epoch"],
                   color="grey", linestyle="--", linewidth=0.8, label="best val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked MSE Loss")
        ax.legend()

    fig.suptitle("Transformer SSL Training")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(args):
    set_seed(args.seed)

    # ── Setup ────────────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    tqdm.write(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    tqdm.write("── Load ──")
    df = load_proteomics(args.input)
    tqdm.write(f"  Loaded: {df.shape[0]} proteins × {df.shape[1]} patients")

    # ── Split first ──────────────────────────────────────────────────────────
    dataset_cls = (
        ProteinTransformerTwoViewDataset
        if args.use_consistency_loss
        else ProteinTransformerDataset
    )

    n_total = len(df)
    n_val = int(n_total * args.val_fraction)

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(args.seed),
    ).tolist()

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    tqdm.write(
        f"  Train proteins: {train_df.shape[0]} | "
        f"Val proteins: {val_df.shape[0]}"
    )

    # ── Compute optional patient-wise normalization from train set only ──────
    means = None
    stds = None
    means_path = checkpoint_dir / "ssl_transformer_patient_means.npy"
    stds_path = checkpoint_dir / "ssl_transformer_patient_stds.npy"

    if args.normalization_mode == "patient":
        means, stds = compute_patient_normalization(train_df)
        np.save(means_path, means)
        np.save(stds_path, stds)
    elif args.normalization_mode not in {"protein", "none"}:
        raise ValueError("normalization_mode must be one of: protein, patient, none")

    # ── Build datasets using train-derived normalization ─────────────────────
    train_dataset = dataset_cls(
        train_df,
        mask_prob=args.mask_prob,
        means=means,
        stds=stds,
        normalization_mode=args.normalization_mode,
        training=True,
        seed=args.seed,
    )

    val_dataset = dataset_cls(
        val_df,
        mask_prob=args.mask_prob,
        means=means,
        stds=stds,
        normalization_mode=args.normalization_mode,
        training=False,
        seed=args.seed + 1,
    )

    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    # ── Model ────────────────────────────────────────────────────────────────
    model = PatientTokenTransformer(
        n_patients=df.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
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
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_path         = checkpoint_dir / "ssl_transformer_best.pt"
    last_path         = checkpoint_dir / "ssl_transformer_last.pt"
    train_log         = []

    def save_checkpoint(path):
        torch.save({
            "model_state_dict":   model.state_dict(),
            "n_patients":         df.shape[1],
            "d_model":            args.d_model,
            "n_heads":            args.n_heads,
            "n_layers":           args.n_layers,
            "dim_feedforward":    args.dim_feedforward,
            "latent_dim":         args.latent_dim,
            "dropout":            args.dropout,
            "mask_prob":          args.mask_prob,
            "use_consistency_loss": args.use_consistency_loss,
            "lambda_consistency": args.lambda_consistency,
            "normalization_mode": args.normalization_mode,
            "patient_means":      torch.from_numpy(means) if means is not None else None,
            "patient_stds":       torch.from_numpy(stds) if stds is not None else None,
            "patient_means_path": str(means_path),
            "patient_stds_path":  str(stds_path),
            "seed":               args.seed,
            "val_fraction":       args.val_fraction,
            "train_indices":      train_idx,
            "val_indices":        val_idx,
        }, path)

    tqdm.write("── Training ──")
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")

    for epoch in epoch_pbar:
        if args.use_consistency_loss:
            train_metrics = train_one_epoch_two_view(
                model, train_loader, optimizer, device, epoch, args.epochs,
                lambda_consistency=args.lambda_consistency,
            )
            val_metrics = evaluate_two_view(
                model, val_loader, device, epoch, args.epochs,
                lambda_consistency=args.lambda_consistency,
            )
            train_loss = train_metrics["loss"]
            val_loss   = val_metrics["loss"]
            selection_loss = val_metrics["recon_loss"]
            row = {"epoch": epoch,
                   "train_loss": train_metrics["loss"],
                   "train_recon_loss": train_metrics["recon_loss"],
                   "train_consistency_loss": train_metrics["consistency_loss"],
                   "val_loss": val_metrics["loss"],
                   "val_recon_loss": val_metrics["recon_loss"],
                   "val_consistency_loss": val_metrics["consistency_loss"]}
        else:
            train_loss = train_one_epoch_single_view(
                model, train_loader, optimizer, device, epoch, args.epochs
            )
            val_loss = evaluate_single_view(
                model, val_loader, device, epoch, args.epochs
            )
            selection_loss = val_loss
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

        scheduler.step()
        train_log.append(row)
        epoch_pbar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}",
                               best=f"{best_val_loss:.6f}")

        if selection_loss < best_val_loss:
            best_val_loss     = selection_loss
            epochs_no_improve = 0
            save_checkpoint(best_path)
        else:
            epochs_no_improve += 1

        if args.patience and epochs_no_improve >= args.patience:
            tqdm.write(f"  Early stopping at epoch {epoch} "
                       f"(no improvement for {args.patience} epochs)")
            break

    save_checkpoint(last_path)

    # ── Save logs, plot, config ───────────────────────────────────────────────
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(checkpoint_dir / "ssl_transformer_train_log.csv", index=False)

    plot_path = checkpoint_dir / "transformer_ssl_loss.png"
    plot_loss(log_df, args.use_consistency_loss, plot_path)
    tqdm.write(f"  Loss plot saved to {plot_path}")

    config = vars(args)
    config.update({"n_proteins": df.shape[0], "n_patients": df.shape[1],
                   "best_val_loss": best_val_loss})
    with open(checkpoint_dir / "transformer_ssl_config.json", "w") as f:
        json.dump(config, f, indent=2)

    tqdm.write(f"  Best val loss:   {best_val_loss:.6f}")
    tqdm.write(f"  Best checkpoint: {best_path}")
    tqdm.write(f"  Last checkpoint: {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",              type=str,   default="data/processed/proteomics_data_processed.csv")
    parser.add_argument("--checkpoint_dir",     type=str,   default="checkpoints/transformer_ssl")
    parser.add_argument("--mask_prob",          type=float, default=0.15)
    parser.add_argument("--d_model",            type=int,   default=64)
    parser.add_argument("--n_heads",            type=int,   default=4)
    parser.add_argument("--n_layers",           type=int,   default=2)
    parser.add_argument("--dim_feedforward",    type=int,   default=128)
    parser.add_argument("--latent_dim",         type=int,   default=32)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--normalization_mode", type=str,   default="protein",
                        choices=["protein", "patient", "none"],
                        help=(
                            "protein: z-score each protein across patients; "
                            "patient: z-score each patient across train proteins; "
                            "none: use input values as-is."
                        ))
    parser.add_argument("--use_consistency_loss", action="store_true")
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--batch_size",         type=int,   default=128)
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--weight_decay",       type=float, default=1e-4)
    parser.add_argument("--val_fraction",       type=float, default=0.1)
    parser.add_argument("--num_workers",        type=int,   default=0)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--patience",           type=int,   default=None,
                        help="Early stopping patience. None = disabled.")
    parser.add_argument("--cpu",                action="store_true")

    args = parser.parse_args()
    main(args)
