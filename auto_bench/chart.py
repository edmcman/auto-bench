from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluator import parse_results


def plot_kl_vs_resolves(results: list[dict], output_path: Path) -> None:
    groups: dict[int, list[float]] = defaultdict(list)
    for r in results:
        if r.get("kl_divergence") is None:
            continue
        resolved, _, _ = parse_results(r.get("results", {}))
        groups[resolved].append(r["kl_divergence"])

    if not groups:
        raise ValueError("No entries with KL divergence data")

    xs = sorted(groups)
    kl_by_resolved = [groups[x] for x in xs]

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(kl_by_resolved, positions=xs, widths=0.6, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        rng = np.random.default_rng(0)
        for x, kls in zip(xs, kl_by_resolved):
            jitter = rng.uniform(-0.08, 0.08, len(kls))
            ax.scatter(x + jitter, kls, s=30, zorder=3, color="white", edgecolors="gray", linewidths=0.8)
        ax.set_xlabel("Resolved")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence by Resolve Count")
        ax.set_xticks(xs)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
