from pathlib import Path

import matplotlib.pyplot as plt

from .evaluator import parse_results


def _strip_common_prefix(names: list[str]) -> list[str]:
    if not names:
        return names
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix.rstrip("-")
    return [n[len(prefix):].lstrip("-") if n.startswith(prefix) else n for n in names]


def plot_kl_vs_resolves(results: list[dict], output_path: Path) -> None:
    kl_entries = [r for r in results if r.get("kl_divergence") is not None]
    ref_entries = [r for r in results if r.get("kl_divergence") is None and r.get("results")]

    if not kl_entries:
        raise ValueError("No entries with KL divergence data")

    all_names = [r.get("name", "") for r in results]
    short = dict(zip(all_names, _strip_common_prefix(all_names)))

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(10, 6))

        xs = [r["kl_divergence"] for r in kl_entries]
        ys = [parse_results(r.get("results", {}))[2] for r in kl_entries]
        labels = [short.get(r.get("name", ""), r.get("name", "")) for r in kl_entries]

        ax.scatter(xs, ys, s=60, zorder=3, color="steelblue", edgecolors="white", linewidths=0.8)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)

        ref_styles = [
            {"color": "firebrick", "linestyle": "--"},
            {"color": "darkorange", "linestyle": "-."},
            {"color": "purple", "linestyle": ":"},
        ]
        for i, r in enumerate(ref_entries):
            _, _, pct = parse_results(r.get("results", {}))
            name = short.get(r.get("name", ""), r.get("name", ""))
            style = ref_styles[i % len(ref_styles)]
            ax.axhline(y=pct, linewidth=1.5, alpha=0.9,
                       label=f"{name} (reference, {pct:.1f}%)", **style)

        ax.set_xscale("log")
        ax.set_xlabel("KL Divergence (log scale)")
        ax.set_ylabel("% Resolved")
        ax.set_title("KL Divergence vs. Resolve Rate")
        if ref_entries:
            ax.legend(fontsize=11, facecolor="white", framealpha=1.0, edgecolor="gray",
                      loc="upper right", borderpad=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
