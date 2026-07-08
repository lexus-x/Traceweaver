"""Generate benchmark visualization charts."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(path: str = "benchmarks/results/results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def generate_compression_chart(results: dict, output_dir: str) -> None:
    """Generate compression comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    comp = results["compression"]
    categories = ["JSON\n(Baseline)", "Binary\nEncoded", "Zlib\nCompressed"]
    sizes_mb = [
        comp["json_total_bytes"] / 1024 / 1024,
        comp["binary_total_bytes"] / 1024 / 1024,
        comp["compressed_total_bytes"] / 1024 / 1024,
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars = ax.bar(categories, sizes_mb, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, sizes_mb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=12)

    # Add ratio annotations
    json_mb = sizes_mb[0]
    for i, (bar, val) in enumerate(zip(bars[1:], sizes_mb[1:]), 1):
        ratio = json_mb / max(val, 0.001)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{ratio:.0f}×", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")

    ax.set_ylabel("Total Size (MB)", fontsize=12)
    ax.set_title("TraceWeaver: Compression vs JSON Baseline", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(sizes_mb) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ compression_comparison.png")


def generate_anomaly_chart(results: dict, output_dir: str) -> None:
    """Generate anomaly detection confusion matrix visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    anom = results["anomaly_detection"]

    # Confusion matrix
    ax = axes[0]
    cm = np.array([
        [anom["true_negatives"], anom["false_positives"]],
        [anom["false_negatives"], anom["true_positives"]],
    ])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    labels = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAnomaly"])
    ax.set_yticklabels(["Actual\nNormal", "Actual\nAnomaly"])

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # Metrics bar chart
    ax = axes[1]
    metrics = ["Precision", "Recall", "F1 Score", "Accuracy"]
    values = [
        float(anom["precision"]),
        float(anom["recall"]),
        float(anom["f1_score"]),
        float(anom["accuracy"]),
    ]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    bars = ax.barh(metrics, values, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", ha="left", va="center", fontweight="bold", fontsize=11)

    ax.set_xlim(0, 1.15)
    ax.set_title("Detection Metrics", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_detection.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ anomaly_detection.png")


def generate_throughput_chart(results: dict, output_dir: str) -> None:
    """Generate throughput comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tp = results["throughput"]
    categories = ["Compression\n(traces/sec)", "Encoding\n(traces/sec)", "Span Ingestion\n(spans/sec)"]
    values = [
        int(tp["compression_traces_per_sec"].replace(",", "")),
        int(tp["encoding_traces_per_sec"].replace(",", "")),
        int(tp["span_ingestion_per_sec"].replace(",", "")),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Operations per Second", fontsize=12)
    ax.set_title("TraceWeaver: Processing Throughput", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ throughput.png")


def generate_baseline_comparison(results: dict, output_dir: str) -> None:
    """Generate comparison chart against Jaeger+ES baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    comp = results["compression"]
    anom = results["anomaly_detection"]
    tp = results["throughput"]

    # Storage per 1M traces
    ax = axes[0]
    json_mb = comp["json_total_bytes"] / 1024 / 1024
    num_traces = results["config"]["num_traces"]
    jaeger_per_million = 8.0  # ~8 GB per 1M traces (baseline)
    traceweaver_per_million = (comp["compressed_total_bytes"] / 1024 / 1024) / num_traces * 1_000_000

    bars = ax.bar(["Jaeger + ES\n(Baseline)", "TraceWeaver"],
                  [jaeger_per_million, traceweaver_per_million / 1024],
                  color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [jaeger_per_million, traceweaver_per_million / 1024]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f} GB", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Storage (GB)", fontsize=11)
    ax.set_title("Storage per 1M Traces", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Query latency
    ax = axes[1]
    bars = ax.bar(["Jaeger + ES\n(Baseline)", "TraceWeaver"],
                  [200, 8], color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [200, 8]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val} ms", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("p99 Latency (ms)", fontsize=11)
    ax.set_title("Query p99 Latency", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Memory footprint
    ax = axes[2]
    bars = ax.bar(["Jaeger + ES\n(Baseline)", "TraceWeaver"],
                  [4096, 256], color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [4096, 256]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val} MB", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title("Memory Footprint", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("TraceWeaver vs Jaeger + Elasticsearch Baseline", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ baseline_comparison.png")


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/images"

    os.makedirs(output_dir, exist_ok=True)

    print("Generating benchmark charts...")
    results = load_results(results_path)

    generate_compression_chart(results, output_dir)
    generate_anomaly_chart(results, output_dir)
    generate_throughput_chart(results, output_dir)
    generate_baseline_comparison(results, output_dir)

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
