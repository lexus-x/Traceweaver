"""CLI for TraceBench — synthetic trace generator and benchmark runner."""

from __future__ import annotations

import json
import os
import sys
import time

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from traceweaver.anomaly import AnomalyDetector, TDigest
from traceweaver.compressor import TraceCompressor
from traceweaver.encoder import TraceEncoder, estimate_json_size
from tracebench.generator import TraceGenerator

console = Console()


@click.group()
def main():
    """TraceBench — Distributed trace compression & anomaly detection benchmarks."""
    pass


@main.command()
@click.option("-n", "--num-traces", default=1000, help="Number of traces to generate")
@click.option("--anomaly-rate", default=0.01, help="Fraction of anomalous traces")
@click.option("-o", "--output", default=None, help="Output file (default: stdout)")
def generate(num_traces: int, anomaly_rate: float, output: str | None):
    """Generate synthetic traces."""
    gen = TraceGenerator(anomaly_rate=anomaly_rate)

    traces = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating traces...", total=num_traces)
        for _ in range(num_traces):
            traces.append(gen.generate_trace())
            progress.advance(task)

    # Summary
    total_spans = sum(len(t.spans) for t in traces)
    avg_spans = total_spans / len(traces) if traces else 0
    avg_duration_ms = np.mean([t.duration_us / 1000 for t in traces]) if traces else 0

    console.print(Panel.fit(
        f"[green]Generated {num_traces} traces[/green]\n"
        f"Total spans: {total_spans:,}\n"
        f"Avg spans/trace: {avg_spans:.1f}\n"
        f"Avg trace duration: {avg_duration_ms:.1f}ms",
        title="TraceBench",
    ))

    if output:
        data = [
            {
                "trace_id": t.trace_id,
                "span_count": len(t.spans),
                "duration_us": t.duration_us,
                "services": list(t.services),
            }
            for t in traces
        ]
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[dim]Saved to {output}[/dim]")


@main.command()
@click.option("-n", "--num-traces", default=5000, help="Number of traces to benchmark")
@click.option("--anomaly-rate", default=0.01, help="Fraction of anomalous traces")
@click.option("-o", "--output", default="benchmarks/results/results.json", help="Results output file")
def benchmark(num_traces: int, anomaly_rate: float, output: str):
    """Run full compression and anomaly detection benchmarks."""
    console.print(Panel.fit(
        "[bold cyan]TraceWeaver Benchmark Suite[/bold cyan]\n"
        f"Traces: {num_traces:,} | Anomaly rate: {anomaly_rate:.1%}",
        title="🏃 Starting Benchmarks",
    ))

    gen = TraceGenerator(anomaly_rate=anomaly_rate)
    compressor = TraceCompressor()
    encoder = TraceEncoder()
    detector = AnomalyDetector(z_threshold=3.0)

    results = {
        "config": {
            "num_traces": num_traces,
            "anomaly_rate": anomaly_rate,
        },
        "compression": {},
        "anomaly_detection": {},
        "throughput": {},
    }

    # Phase 1: Generate traces
    console.print("\n[bold]Phase 1:[/bold] Generating synthetic traces...")
    traces = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=num_traces)
        for _ in range(num_traces):
            traces.append(gen.generate_trace())
            progress.advance(task)

    total_spans = sum(len(t.spans) for t in traces)
    console.print(f"  ✓ Generated {num_traces:,} traces with {total_spans:,} total spans")

    # Phase 2: Compression benchmark
    console.print("\n[bold]Phase 2:[/bold] Compression benchmark...")

    json_total = 0
    binary_total = 0
    compressed_total = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Compressing...", total=num_traces)
        for trace in traces:
            # JSON baseline
            json_size = estimate_json_size(trace)
            json_total += json_size

            # Our binary encoder
            binary = encoder.encode(trace)
            binary_total += len(binary)

            # Compressed
            compressed = encoder.encode_compressed(trace)
            compressed_total += len(compressed)

            # Template-based compression
            compressor.compress(trace)

            progress.advance(task)

    results["compression"] = {
        "json_total_bytes": json_total,
        "binary_total_bytes": binary_total,
        "compressed_total_bytes": compressed_total,
        "binary_vs_json_ratio": f"{json_total / max(binary_total, 1):.1f}x",
        "compressed_vs_json_ratio": f"{json_total / max(compressed_total, 1):.1f}x",
        "template_compression_ratio": f"{compressor.overall_compression_ratio:.1f}x",
        "template_hit_rate": f"{compressor.template_hit_rate:.1%}",
        "templates_learned": len(compressor.templates),
    }

    # Phase 3: Anomaly detection benchmark
    console.print("\n[bold]Phase 3:[/bold] Anomaly detection benchmark...")

    # First, train on normal traces
    normal_traces, anomaly_traces = gen.generate_dataset(
        normal_count=min(num_traces, 2000),
        anomaly_count=max(1, int(num_traces * anomaly_rate)),
    )

    console.print(f"  Training on {len(normal_traces):,} normal traces...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training detector...", total=len(normal_traces))
        for trace in normal_traces:
            detector.update(trace)
            progress.advance(task)

    # Then, score all traces
    console.print(f"  Scoring {len(anomaly_traces):,} anomalous + {len(normal_traces):,} normal traces...")

    tp = fp = tn = fn = 0
    scores = []

    all_traces = [(t, False) for t in normal_traces] + [(t, True) for t in anomaly_traces]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scoring...", total=len(all_traces))
        for trace, is_actual_anomaly in all_traces:
            score = detector.score(trace)
            scores.append(score)

            if score.is_anomaly and is_actual_anomaly:
                tp += 1
            elif score.is_anomaly and not is_actual_anomaly:
                fp += 1
            elif not score.is_anomaly and not is_actual_anomaly:
                tn += 1
            else:
                fn += 1

            progress.advance(task)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    results["anomaly_detection"] = {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": f"{precision:.3f}",
        "recall": f"{recall:.3f}",
        "f1_score": f"{f1:.3f}",
        "accuracy": f"{accuracy:.3f}",
    }

    # Phase 4: Throughput benchmark
    console.print("\n[bold]Phase 4:[/bold] Throughput benchmark...")

    # Compression throughput
    start = time.perf_counter()
    for trace in traces[:1000]:
        compressor.compress(trace)
    elapsed = time.perf_counter() - start
    compress_throughput = 1000 / max(elapsed, 0.001)

    # Encoding throughput
    start = time.perf_counter()
    for trace in traces[:1000]:
        encoder.encode(trace)
    elapsed = time.perf_counter() - start
    encode_throughput = 1000 / max(elapsed, 0.001)

    # Span ingestion throughput
    start = time.perf_counter()
    span_count = 0
    for trace in traces[:1000]:
        span_count += len(trace.spans)
        detector.update(trace)
    elapsed = time.perf_counter() - start
    span_throughput = span_count / max(elapsed, 0.001)

    results["throughput"] = {
        "compression_traces_per_sec": f"{compress_throughput:,.0f}",
        "encoding_traces_per_sec": f"{encode_throughput:,.0f}",
        "span_ingestion_per_sec": f"{span_throughput:,.0f}",
    }

    # Display results
    _display_results(results)

    # Save results
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to {output}[/dim]")


def _display_results(results: dict) -> None:
    """Display benchmark results in a rich table."""
    console.print("\n")

    # Compression results
    table = Table(title="📊 Compression Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    comp = results["compression"]
    table.add_row("JSON total size", f"{comp['json_total_bytes'] / 1024 / 1024:.1f} MB")
    table.add_row("Binary encoded", f"{comp['binary_total_bytes'] / 1024 / 1024:.1f} MB")
    table.add_row("Compressed (zlib)", f"{comp['compressed_total_bytes'] / 1024 / 1024:.1f} MB")
    table.add_row("Binary vs JSON", comp["binary_vs_json_ratio"])
    table.add_row("Compressed vs JSON", comp["compressed_vs_json_ratio"])
    table.add_row("Template compression", comp["template_compression_ratio"])
    table.add_row("Template hit rate", comp["template_hit_rate"])
    table.add_row("Templates learned", str(comp["templates_learned"]))
    console.print(table)

    # Anomaly detection results
    table = Table(title="🎯 Anomaly Detection Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    anom = results["anomaly_detection"]
    table.add_row("True Positives", str(anom["true_positives"]))
    table.add_row("False Positives", str(anom["false_positives"]))
    table.add_row("True Negatives", str(anom["true_negatives"]))
    table.add_row("False Negatives", str(anom["false_negatives"]))
    table.add_row("Precision", anom["precision"])
    table.add_row("Recall", anom["recall"])
    table.add_row("F1 Score", anom["f1_score"])
    table.add_row("Accuracy", anom["accuracy"])
    console.print(table)

    # Throughput results
    table = Table(title="⚡ Throughput Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    tp = results["throughput"]
    table.add_row("Compression", f"{tp['compression_traces_per_sec']} traces/sec")
    table.add_row("Encoding", f"{tp['encoding_traces_per_sec']} traces/sec")
    table.add_row("Span ingestion", f"{tp['span_ingestion_per_sec']} spans/sec")
    console.print(table)


if __name__ == "__main__":
    main()
