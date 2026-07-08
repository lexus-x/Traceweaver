"""CLI for TraceWeaver — trace compression and anomaly detection."""

from __future__ import annotations

import json
import os
import time

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .anomaly import AnomalyDetector
from .compressor import TraceCompressor
from .encoder import TraceEncoder, estimate_json_size

console = Console()


@click.group()
def main():
    """TraceWeaver — Intelligent distributed trace compression & anomaly detection."""
    pass


@main.command()
@click.argument("traces_file")
@click.option("-o", "--output", default=None, help="Output file")
def compress(traces_file: str, output: str | None):
    """Compress traces from a JSON file."""
    with open(traces_file) as f:
        data = json.load(f)

    console.print(f"[bold]Loaded {len(data)} traces[/bold]")

    compressor = TraceCompressor()
    encoder = TraceEncoder()

    for item in data:
        from traceweaver.trace import Trace, Span, SpanStatus
        trace = Trace(trace_id=item["trace_id"])
        for s in item.get("spans", []):
            trace.spans.append(Span(
                trace_id=item["trace_id"],
                span_id=s["span_id"],
                parent_id=s.get("parent_id"),
                service_name=s["service"],
                operation_name=s["operation"],
                start_time_us=s["start_us"],
                duration_us=s["duration_us"],
                status=SpanStatus(s.get("status", 0)),
            ))
        compressor.compress(trace)

    stats = compressor.stats()
    console.print(Panel.fit(
        f"Compressed: {stats['traces_compressed']} traces\n"
        f"Ratio: {stats['compression_ratio']}\n"
        f"Template hit rate: {stats['template_hit_rate']}\n"
        f"Templates: {stats['templates_learned']}",
        title="Compression Results",
    ))


@main.command()
def version():
    """Show version."""
    from . import __version__
    console.print(f"TraceWeaver v{__version__}")


if __name__ == "__main__":
    main()
