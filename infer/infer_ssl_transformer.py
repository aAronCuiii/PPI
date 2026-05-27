# src/inference/infer_ssl_transformer.py

"""
Infer protein embeddings from a trained SSL transformer and compute cosine similarities.

python -m infer.infer_ssl_transformer \
  --input data/processed/proteomics_data_processed.csv \
  --checkpoint checkpoints/ssl_transformer/ssl_transformer_best.pt \
  --output_dir outputs/similarity_transformer \
  --normalization_mode checkpoint \
  --stats_mode checkpoint \
  --batch_size 512 \
  --top_k 100000
"""

import argparse
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.ssl_transformer import PatientTokenTransformer


# ============================================================
# Data loading
# ============================================================

def load_proteomics(path: str) -> pd.DataFrame:
    """
    Load protein x patient proteomics matrix.

    Rows:
        proteins

    Columns:
        patients

    Values:
        normalized log2 abundance values, with NaN for not quantified
    """
    try:
        df = pd.read_csv(path, index_col=0)

        if df.shape[1] <= 1:
            df = pd.read_csv(path, index_col=0, sep="\t")

    except Exception:
        df = pd.read_csv(path, index_col=0, sep="\t")

    return df


def compute_normalization_stats(
    df: pd.DataFrame,
    mode: str = "train_split",
    seed: int = 42,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute patient-wise normalization statistics.

    mode = "checkpoint":
        Use normalization statistics saved in the checkpoint. This is the
        safest option because cosine embeddings depend on train-time scaling.

    mode = "train_split":
        Recreate the same train/validation split used during training.
        Compute patient-wise means and stds from the training proteins only.

    mode = "full":
        Compute patient-wise means and stds from all proteins.
        This is less consistent with train-only normalization, but useful
        if the original training split information is unavailable.
    """
    if mode == "train_split":
        n_total = len(df)
        n_val = int(n_total * val_fraction)

        indices = torch.randperm(
            n_total,
            generator=torch.Generator().manual_seed(seed),
        ).tolist()

        train_idx = indices[n_val:]
        stat_df = df.iloc[train_idx].copy()

        print(
            f"Normalization mode: train_split | "
            f"train proteins used for stats: {len(stat_df)}"
        )

    elif mode == "full":
        stat_df = df
        print(f"Normalization mode: full | proteins used for stats: {len(stat_df)}")

    else:
        raise ValueError(
            f"Unknown stats mode: {mode}. "
            "Choose from: checkpoint, train_split, or full"
        )

    X_raw = stat_df.values.astype(np.float32)

    means = np.nanmean(X_raw, axis=0).astype(np.float32)
    stds = np.nanstd(X_raw, axis=0).astype(np.float32)

    stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds).astype(np.float32)

    return means, stds


def normalization_stats_from_checkpoint(
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    if "patient_means" in checkpoint and "patient_stds" in checkpoint:
        means = checkpoint["patient_means"]
        stds = checkpoint["patient_stds"]

        if means is None or stds is None:
            return None

        if torch.is_tensor(means):
            means = means.detach().cpu().numpy()
        if torch.is_tensor(stds):
            stds = stds.detach().cpu().numpy()

        return means.astype(np.float32), stds.astype(np.float32)

    means_path = checkpoint.get("patient_means_path")
    stds_path = checkpoint.get("patient_stds_path")
    if means_path and stds_path and Path(means_path).exists() and Path(stds_path).exists():
        return (
            np.load(means_path).astype(np.float32),
            np.load(stds_path).astype(np.float32),
        )

    return None


# ============================================================
# Inference dataset
# ============================================================

class ProteinTransformerInferenceDataset(Dataset):
    """
    Dataset for transformer inference.

    During inference:
        no artificial masking is applied

    x:
        standardized abundance values with natural NaNs filled by 0

    obs_mask:
        original detection mask, 1 if quantified, 0 if naturally missing

    ssl_mask:
        all zeros during inference
    """

    def __init__(
        self,
        df: pd.DataFrame,
        means: np.ndarray | None,
        stds: np.ndarray | None,
        normalization_mode: str,
    ):
        self.proteins = df.index.astype(str).tolist()

        X_raw = df.values.astype(np.float32)

        self.obs_mask = (~np.isnan(X_raw)).astype(np.float32)
        self.normalization_mode = normalization_mode

        if normalization_mode == "patient":
            if means is None or stds is None:
                raise ValueError("patient normalization requires means and stds.")
            means = means.astype(np.float32)
            stds = stds.astype(np.float32)
            stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds).astype(np.float32)
            X_std = (X_raw - means) / stds
        elif normalization_mode == "protein":
            row_means = np.nanmean(X_raw, axis=1, keepdims=True)
            row_stds = np.nanstd(X_raw, axis=1, keepdims=True)
            row_stds = np.where((row_stds == 0) | np.isnan(row_stds), 1.0, row_stds)
            X_std = (X_raw - row_means) / row_stds
        elif normalization_mode == "none":
            X_std = X_raw.copy()
        else:
            raise ValueError(
                f"Unknown normalization_mode={normalization_mode}. "
                "Choose from: protein, patient, none."
            )
        self.X_filled = np.where(np.isnan(X_std), 0.0, X_std).astype(np.float32)

        self.ssl_mask = np.zeros_like(self.X_filled, dtype=np.float32)

    def __len__(self):
        return self.X_filled.shape[0]

    def __getitem__(self, idx):
        x = self.X_filled[idx]
        obs = self.obs_mask[idx]
        ssl = self.ssl_mask[idx]
        protein = self.proteins[idx]

        return (
            torch.from_numpy(x),
            torch.from_numpy(obs),
            torch.from_numpy(ssl),
            protein,
        )


# ============================================================
# Model loading
# ============================================================

def load_checkpoint(path: str, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model_state_dict. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> PatientTokenTransformer:
    """
    Rebuild transformer architecture from checkpoint metadata.
    """
    required_keys = [
        "n_patients",
        "d_model",
        "n_heads",
        "n_layers",
        "dim_feedforward",
        "latent_dim",
        "dropout",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")

    model = PatientTokenTransformer(
        n_patients=int(checkpoint["n_patients"]),
        d_model=int(checkpoint["d_model"]),
        n_heads=int(checkpoint["n_heads"]),
        n_layers=int(checkpoint["n_layers"]),
        dim_feedforward=int(checkpoint["dim_feedforward"]),
        latent_dim=int(checkpoint["latent_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# ============================================================
# Embedding extraction
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model: PatientTokenTransformer,
    loader: DataLoader,
    device: torch.device,
    use_three_channels: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Extract one latent embedding per protein.
    """
    all_proteins = []
    all_embeddings = []

    for x, obs, ssl_mask, proteins in tqdm(loader, desc="Extract embeddings"):
        x = x.to(device)
        obs = obs.to(device)
        ssl_mask = ssl_mask.to(device)

        if use_three_channels:
            _, z = model(x, obs, ssl_mask)
        else:
            _, z = model(x, obs)

        all_embeddings.append(z.detach().cpu().numpy().astype(np.float32))
        all_proteins.extend(list(proteins))

    Z = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    return all_proteins, Z


# ============================================================
# Cosine similarity
# ============================================================

def cosine_similarity_matrix(Z: np.ndarray) -> np.ndarray:
    """
    Compute full protein x protein cosine similarity matrix.
    """
    Z = Z.astype(np.float32)

    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)

    Z_norm = Z / norms

    sim = Z_norm @ Z_norm.T
    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)

    np.fill_diagonal(sim, 1.0)

    return sim


def similarity_matrix_to_edge_table(
    sim: np.ndarray,
    proteins: list[str],
    top_k: int | None = 100000,
    min_score: float | None = None,
) -> pd.DataFrame:
    """
    Convert full cosine similarity matrix to ranked edge table.
    """
    if top_k is None:
        rows, cols = np.triu_indices_from(sim, k=1)
        scores = sim[rows, cols]

        mask = ~np.isnan(scores)

        if min_score is not None:
            mask &= scores >= min_score

        edge_df = pd.DataFrame({
            "protein_1": np.array(proteins)[rows[mask]],
            "protein_2": np.array(proteins)[cols[mask]],
            "transformer_cosine": scores[mask],
        })
    else:
        heap: list[tuple[float, int, int]] = []
        n = sim.shape[0]

        for i in range(n - 1):
            row_scores = sim[i, i + 1 :]
            valid = np.isfinite(row_scores)
            if min_score is not None:
                valid &= row_scores >= min_score
            if len(heap) >= top_k:
                valid &= row_scores > heap[0][0]

            offsets = np.flatnonzero(valid)
            if offsets.size == 0:
                continue

            candidate_scores = row_scores[offsets]
            row_keep = min(offsets.size, top_k)
            if row_keep < offsets.size:
                keep = np.argpartition(candidate_scores, -row_keep)[-row_keep:]
                offsets = offsets[keep]
                candidate_scores = candidate_scores[keep]

            for offset, score in zip(offsets, candidate_scores, strict=False):
                item = (float(score), i, i + int(offset) + 1)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        rows = sorted(heap, reverse=True)
        edge_df = pd.DataFrame({
            "protein_1": [proteins[i] for _, i, _ in rows],
            "protein_2": [proteins[j] for _, _, j in rows],
            "transformer_cosine": [score for score, _, _ in rows],
        })

    edge_df = edge_df.sort_values(
        "transformer_cosine",
        ascending=False,
    ).reset_index(drop=True)

    edge_df["rank"] = np.arange(1, len(edge_df) + 1)

    if top_k is not None:
        edge_df = edge_df.head(top_k).copy()

    return edge_df


# ============================================================
# Main
# ============================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Device: {device}")

    print("── Load data ──")
    df = load_proteomics(args.input)
    print(f"Loaded data: {df.shape[0]} proteins x {df.shape[1]} patients")

    print("── Load checkpoint ──")
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    print("── Load normalization stats ──")
    normalization_mode = args.normalization_mode
    if normalization_mode == "checkpoint":
        normalization_mode = checkpoint.get("normalization_mode", "patient")
    if normalization_mode not in {"protein", "patient", "none"}:
        raise ValueError(
            "normalization_mode must be one of: checkpoint, protein, patient, none"
        )

    means = None
    stds = None
    if normalization_mode == "patient" and args.stats_mode == "checkpoint":
        stats = normalization_stats_from_checkpoint(checkpoint)
        if stats is None:
            raise ValueError(
                "stats_mode=checkpoint requested, but this checkpoint does not "
                "contain patient_means/patient_stds or readable stats paths."
            )
        means, stds = stats
        print("Normalization mode: patient checkpoint stats")
    elif normalization_mode == "patient":
        means, stds = compute_normalization_stats(
            df=df,
            mode=args.stats_mode,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
    else:
        print(f"Normalization mode: {normalization_mode}")

    if normalization_mode == "patient" and (
        len(means) != df.shape[1] or len(stds) != df.shape[1]
    ):
        raise ValueError(
            f"Normalization length mismatch: "
            f"means={len(means)}, stds={len(stds)}, patients={df.shape[1]}"
        )

    print("── Build model ──")
    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    print("── Build inference dataset ──")
    dataset = ProteinTransformerInferenceDataset(
        df=df,
        means=means,
        stds=stds,
        normalization_mode=normalization_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("── Extract embeddings ──")
    proteins, Z = extract_embeddings(
        model=model,
        loader=loader,
        device=device,
        use_three_channels=args.use_three_channels,
    )

    print(f"Embeddings shape: {Z.shape}")

    print("── Save embeddings ──")
    emb_cols = [f"z_{i + 1}" for i in range(Z.shape[1])]
    emb_df = pd.DataFrame(Z, index=proteins, columns=emb_cols)
    emb_df.index.name = "protein"
    emb_df.to_csv(output_dir / "transformer_embeddings.csv")

    pd.Series(proteins, name="protein").to_csv(
        output_dir / "protein_index.csv",
        index=False,
    )

    print("── Compute cosine similarity ──")
    sim = cosine_similarity_matrix(Z)

    np.save(output_dir / "transformer_cosine.npy", sim)

    vals = sim[np.triu_indices_from(sim, k=1)]

    print("Cosine summary:")
    print(f"  mean: {vals.mean():.4f}")
    print(f"  std:  {vals.std():.4f}")
    print(f"  min:  {vals.min():.4f}")
    print(f"  max:  {vals.max():.4f}")
    print(f"  q95:  {np.quantile(vals, 0.95):.4f}")
    print(f"  q99:  {np.quantile(vals, 0.99):.4f}")

    print("── Save ranked edges ──")
    edge_df = similarity_matrix_to_edge_table(
        sim=sim,
        proteins=proteins,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    edge_path = output_dir / f"transformer_cosine_top{args.top_k}_edges.csv"
    edge_df.to_csv(edge_path, index=False)

    print("── Done ──")
    print(f"Saved outputs to: {output_dir}")
    print(f"Saved edge table: {edge_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/proteomics_data_processed.csv",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ssl_transformer_best.pt or ssl_transformer_last.pt",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/similarity_transformer",
    )

    parser.add_argument(
        "--stats_mode",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "train_split", "full"],
        help="How to recompute patient-wise normalization statistics.",
    )

    parser.add_argument(
        "--normalization_mode",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "protein", "patient", "none"],
        help="Input normalization. checkpoint uses the mode saved during training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to reproduce the training split when stats_mode=train_split.",
    )

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Validation fraction used during training.",
    )

    parser.add_argument(
        "--use_three_channels",
        action="store_true",
        help="Use if model forward is forward(x, obs_mask, ssl_mask).",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
