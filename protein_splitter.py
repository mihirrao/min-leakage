import os

_env_limits = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
for k, v in _env_limits.items():
    os.environ.setdefault(k, v)

import argparse
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import umap

from maxcut_splitter import MaxCutSplitter

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def generate_protein_sequences(n_sequences: int, length: int = 100) -> List[str]:
    return ["".join(np.random.choice(list(AA_ALPHABET), size=length)) for _ in range(n_sequences)]


def load_sequences(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_esm_embeddings(
    sequences: List[str],
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> np.ndarray:
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    outs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Computing embeddings", leave=False):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            hs = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outs.append(pooled.detach().cpu().numpy())

    return np.vstack(outs)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def plot_edge_distributions(
    before_edge_matrix: np.ndarray,
    after_mean_edge_matrix: np.ndarray,
    after_min_edge_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    before_mean: float,
    after_mean_mean: float,
    after_min_mean: float,
    save_path: str = "edge_distributions.png",
):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, hspace=0.3, height_ratios=[1, 1])

    ax_all = fig.add_subplot(gs[0, 0])
    all_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    all_mean = float(all_dist.mean()) if all_dist.size else float("nan")
    ax_all.hist(all_dist, bins=50, alpha=0.7, edgecolor="black", color="grey")
    ax_all.axvline(all_mean, color="black", linestyle="--", linewidth=2, label=f"Mean: {all_mean:.3f}")
    ax_all.set_xlabel("Distance (1 - Cosine Similarity)", fontsize=12)
    ax_all.set_ylabel("Frequency", fontsize=12)
    ax_all.set_title("All Pairwise Distances (Entire Dataset)", fontsize=14, fontweight="bold")
    ax_all.legend()
    ax_all.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    before_dist = before_edge_matrix.flatten()
    after_mean_dist = after_mean_edge_matrix.flatten()
    after_min_dist = after_min_edge_matrix.flatten()

    ax.hist(before_dist, bins=50, alpha=0.45, edgecolor="black", color="blue", label=f"Before (Mean: {before_mean:.3f})")
    ax.hist(after_mean_dist, bins=50, alpha=0.45, edgecolor="black", color="red", label=f"Mean-Max (Mean: {after_mean_mean:.3f})")
    ax.hist(after_min_dist, bins=50, alpha=0.45, edgecolor="black", color="orange", label=f"Min-Max (Mean: {after_min_mean:.3f})")

    ax.axvline(before_mean, linestyle="--", linewidth=2, alpha=0.7, color="blue")
    ax.axvline(after_mean_mean, linestyle="--", linewidth=2, alpha=0.7, color="red")
    ax.axvline(after_min_mean, linestyle="--", linewidth=2, alpha=0.7, color="orange")

    ax.set_xlabel("Distance (1 - Cosine Similarity)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Cross-split Distances: Before vs Mean-Max vs Min-Max", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_optimization_trajectory_two(
    history_mean: List[dict],
    history_min: List[dict],
    desired_test_split_pct: float,
    buffer_pct: float,
    save_path: str = "trajectory.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    lo = desired_test_split_pct - buffer_pct
    hi = desired_test_split_pct + buffer_pct

    # Left panel: Mean-Max trajectory
    ax = axes[0]
    ax.axhspan(lo, hi, alpha=0.15, color="gray", label=f"Allowed range ({lo:.1f}% - {hi:.1f}%)")
    ax.axhline(desired_test_split_pct, color="black", linestyle="--", linewidth=2, label=f"Target: {desired_test_split_pct:.1f}%")
    ax.axhline(lo, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.axhline(hi, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    if history_mean:
        md = [h["mean_distance"] for h in history_mean]
        pcts = [h["test_split_pct"] for h in history_mean]
        ax.plot(md, pcts, linewidth=2, label="Mean-Max trajectory", color="red")
        ax.scatter(md[0], pcts[0], s=90, marker="o", label="Start", color="red", edgecolors="black", linewidth=1)
        ax.scatter(md[-1], pcts[-1], s=90, marker="s", label="End", color="red", edgecolors="black", linewidth=1)

    ax.set_xlabel("Mean cross-split distance", fontsize=12)
    ax.set_ylabel("Test Split Percentage (%)", fontsize=12)
    ax.set_title("Mean-Max Optimization Trajectory", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)

    # Right panel: Min-Max trajectory
    ax = axes[1]
    ax.axhspan(lo, hi, alpha=0.15, color="gray", label=f"Allowed range ({lo:.1f}% - {hi:.1f}%)")
    ax.axhline(desired_test_split_pct, color="black", linestyle="--", linewidth=2, label=f"Target: {desired_test_split_pct:.1f}%")
    ax.axhline(lo, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.axhline(hi, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    min_points = [h for h in history_min if h.get("feasible", False)]
    if min_points:
        mins = [h["min_distance"] for h in min_points]
        pcts2 = [h["test_split_pct"] for h in min_points]
        ax.plot(mins, pcts2, linewidth=2, label="Min-Max trajectory", color="orange")
        ax.scatter(mins[0], pcts2[0], s=90, marker="^", label="First feasible", color="orange", edgecolors="black", linewidth=1)
        ax.scatter(mins[-1], pcts2[-1], s=90, marker="D", label="Best", color="orange", edgecolors="black", linewidth=1)

    ax.set_xlabel("Min cross-split distance", fontsize=12)
    ax.set_ylabel("Test Split Percentage (%)", fontsize=12)
    ax.set_title("Min-Max Optimization Trajectory", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap_embeddings_two(
    embeddings: np.ndarray,
    before_train_val_indices: List[int],
    before_test_indices: List[int],
    after_mean_tv: List[int],
    after_mean_t: List[int],
    after_min_tv: List[int],
    after_min_t: List[int],
    save_path: str = "umap_embeddings.png",
):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=1)
    z = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    a_tv = z[before_train_val_indices]
    a_t = z[before_test_indices]
    ax = axes[0]
    ax.scatter(a_tv[:, 0], a_tv[:, 1], alpha=0.9, s=50, label="Train/Val", color="black")
    ax.scatter(a_t[:, 0], a_t[:, 1], alpha=0.9, s=50, label="Test", color="dodgerblue")
    ax.set_title("Before", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    b_tv = z[after_mean_tv]
    b_t = z[after_mean_t]
    ax = axes[1]
    ax.scatter(b_tv[:, 0], b_tv[:, 1], alpha=0.9, s=50, label="Train/Val", color="black")
    ax.scatter(b_t[:, 0], b_t[:, 1], alpha=0.9, s=50, label="Test", color="dodgerblue")
    ax.set_title("After Mean-Max", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    c_tv = z[after_min_tv]
    c_t = z[after_min_t]
    ax = axes[2]
    ax.scatter(c_tv[:, 0], c_tv[:, 1], alpha=0.9, s=50, label="Train/Val", color="black")
    ax.scatter(c_t[:, 0], c_t[:, 1], alpha=0.9, s=50, label="Test", color="dodgerblue")
    ax.set_title("After Min-Max", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Split protein sequences using ESM embeddings")
    p.add_argument("--sequences", type=str, default="sequences.txt")
    p.add_argument("--n_sequences", type=int, default=200)
    p.add_argument("--sequence_length", type=int, default=100)
    p.add_argument("--desired_test_split_pct", type=float, default=20.0)
    p.add_argument("--buffer_pct", type=float, default=2.0)
    p.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    if not os.path.exists(args.sequences):
        seqs = generate_protein_sequences(args.n_sequences, args.sequence_length)
        with open(args.sequences, "w") as f:
            f.write("\n".join(seqs) + "\n")
        print(f"Generated {len(seqs)} sequences -> {args.sequences}")
    else:
        seqs = load_sequences(args.sequences)

    emb = get_esm_embeddings(seqs, model_name=args.model_name, batch_size=args.batch_size)
    dist = cosine_distance_matrix(emb)

    splitter = MaxCutSplitter(
        distance_matrix=dist,
        desired_test_split_pct=args.desired_test_split_pct,
        buffer_pct=args.buffer_pct,
    )
    splitter.split()

    before_edges, before_stats = splitter.compute_bipartite_statistics(splitter.train_val_indices, splitter.test_indices)

    mean_edges, mean_stats = splitter.compute_bipartite_statistics(
        splitter.best_mean_train_val_indices, splitter.best_mean_test_indices
    )
    min_edges, min_stats = splitter.compute_bipartite_statistics(
        splitter.best_min_train_val_indices, splitter.best_min_test_indices
    )

    plot_edge_distributions(
        before_edges,
        mean_edges,
        min_edges,
        dist,
        before_stats["mean_distance"],
        mean_stats["mean_distance"],
        min_stats["mean_distance"],
    )

    plot_optimization_trajectory_two(
        splitter.history_mean,
        splitter.history_min,
        args.desired_test_split_pct,
        args.buffer_pct,
    )

    plot_umap_embeddings_two(
        emb,
        splitter.train_val_indices,
        splitter.test_indices,
        splitter.best_mean_train_val_indices,
        splitter.best_mean_test_indices,
        splitter.best_min_train_val_indices,
        splitter.best_min_test_indices,
    )

    if splitter.best_min_tau is not None:
        print(f"Min-Max best tau: {splitter.best_min_tau:.6f}")
        print(f"Min-Max achieved min cross distance: {min_stats['min_distance']:.6f}")


if __name__ == "__main__":
    main()