"""DAG-based trace graph compression with template matching."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass, field

import mmh3
import numpy as np

from .trace import Span, Trace


@dataclass
class TraceTemplate:
    """A learned template representing a common trace graph structure."""
    template_id: str
    adjacency: dict[str | None, list[str]]  # parent -> [children]
    service_operations: dict[str, str]  # span_id -> service:operation
    frequency: int = 0
    signature: str = ""

    def __post_init__(self):
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        edges = []
        for parent, children in self.adjacency.items():
            p_label = "root" if parent is None else parent
            for child in children:
                op = self.service_operations.get(child, "?")
                edges.append(f"{p_label}->{child}:{op}")
        edges.sort()
        return "|".join(edges)


@dataclass
class CompressedTrace:
    """A compressed representation of a trace."""
    trace_id: str
    template_id: str | None  # None if no template matched
    deviations: list[dict]  # list of deviations from template
    span_data: bytes  # delta-encoded span timing data
    original_size: int
    compressed_size: int
    is_anomaly: bool = False

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size == 0:
            return float("inf")
        return self.original_size / self.compressed_size


class TraceCompressor:
    """
    DAG-based trace compression engine.

    Compression strategy:
    1. Build structural templates from observed trace patterns
    2. Match incoming traces to templates using MinHash similarity
    3. Store only deviations from the matched template
    4. Delta-encode timing data using varint encoding
    5. Dictionary-encode service and operation names
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.templates: dict[str, TraceTemplate] = {}
        self.service_dict: dict[str, int] = {}  # service -> id
        self.operation_dict: dict[str, int] = {}  # operation -> id
        self._dict_counter = 0

        # Stats
        self.traces_compressed = 0
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        self.template_hits = 0
        self.template_misses = 0

    def _get_dict_id(self, d: dict[str, int], key: str) -> int:
        if key not in d:
            d[key] = self._dict_counter
            self._dict_counter += 1
        return d[key]

    def _trace_to_edges(self, trace: Trace) -> list[tuple[str | None, str, str]]:
        """Extract edges from trace as (parent_id, span_id, service:operation)."""
        edges = []
        for span in trace.spans:
            op = f"{span.service_name}:{span.operation_name}"
            edges.append((span.parent_id, span.span_id, op))
        return sorted(edges, key=lambda x: (x[0] or "", x[1], x[2]))

    def _minhash_signature(self, edges: list[tuple], num_hashes: int = 128) -> np.ndarray:
        """Compute MinHash signature for a set of edges."""
        signature = np.full(num_hashes, np.inf)

        for edge in edges:
            edge_str = f"{edge[0]}->{edge[1]}:{edge[2]}"
            for i in range(num_hashes):
                h = mmh3.hash(edge_str, i, signed=False)
                signature[i] = min(signature[i], h)

        return signature

    def _jaccard_estimate(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        return float(np.mean(sig1 == sig2))

    def _find_template(self, trace: Trace) -> TraceTemplate | None:
        """Find the best matching template for a trace."""
        if not self.templates:
            return None

        # Fast path: use structural signature for exact-ish matching
        edges = self._trace_to_edges(trace)
        trace_ops = frozenset(s.service_operation for s in trace.spans)
        trace_span_count = len(trace.spans)

        best_template = None
        best_score = 0.0

        for template in self.templates.values():
            template_ops = frozenset(template.service_operations.values())
            template_span_count = sum(len(v) for v in template.adjacency.values())

            # Quick size filter
            if abs(trace_span_count - template_span_count) > max(3, trace_span_count * 0.2):
                continue

            # Jaccard on operation sets (fast)
            intersection = len(trace_ops & template_ops)
            union = len(trace_ops | template_ops)
            if union == 0:
                continue
            score = intersection / union

            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= self.similarity_threshold:
            return best_template
        return None

    def _learn_template(self, trace: Trace) -> TraceTemplate:
        """Create a new template from a trace."""
        adjacency = trace.adjacency_list()
        service_ops = {}
        for span in trace.spans:
            service_ops[span.span_id] = f"{span.service_name}:{span.operation_name}"

        # Normalize span IDs to positions for template matching
        span_id_map = {}
        normalized_adj = {}
        normalized_ops = {}

        for i, span in enumerate(trace.spans):
            span_id_map[span.span_id] = f"span_{i}"
            normalized_ops[f"span_{i}"] = f"{span.service_name}:{span.operation_name}"

        normalized_adj[None] = [span_id_map.get(sid, sid) for sid in adjacency.get(None, [])]
        for parent, children in adjacency.items():
            if parent is not None:
                norm_parent = span_id_map.get(parent, parent)
                normalized_adj[norm_parent] = [span_id_map.get(sid, sid) for sid in children]

        template_id = hashlib.md5(
            json.dumps(normalized_adj, default=str).encode()
        ).hexdigest()[:12]

        return TraceTemplate(
            template_id=template_id,
            adjacency=normalized_adj,
            service_operations=normalized_ops,
            frequency=1,
        )

    def _compute_deviations(self, trace: Trace, template: TraceTemplate) -> list[dict]:
        """Compute deviations between a trace and its matched template."""
        deviations = []

        # Check span count difference
        template_span_count = sum(len(v) for v in template.adjacency.values())
        if len(trace.spans) != template_span_count:
            deviations.append({
                "type": "span_count",
                "expected": template_span_count,
                "actual": len(trace.spans),
            })

        # Check service/operation mismatches
        for span in trace.spans:
            expected_op = f"{span.service_name}:{span.operation_name}"
            # Simple deviation: check if this operation exists in template
            if expected_op not in template.service_operations.values():
                deviations.append({
                    "type": "extra_span",
                    "operation": expected_op,
                })

        return deviations

    def _delta_encode_timestamps(self, trace: Trace) -> bytes:
        """Delta-encode span timestamps using varint encoding."""
        if not trace.spans:
            return b""

        # Sort by start time
        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time_us)

        # Delta encode
        encoded = bytearray()
        prev_time = 0

        for span in sorted_spans:
            delta = span.start_time_us - prev_time
            prev_time = span.start_time_us

            # Varint encode the delta
            while delta > 0x7F:
                encoded.append((delta & 0x7F) | 0x80)
                delta >>= 7
            encoded.append(delta & 0x7F)

            # Duration as fixed 4 bytes (microseconds, max ~70 min)
            encoded.extend(struct.pack("<I", min(span.duration_us, 0xFFFFFFFF)))

            # Status byte
            encoded.append(span.status)

        return bytes(encoded)

    def compress(self, trace: Trace) -> CompressedTrace:
        """Compress a trace using template matching and delta encoding."""
        self.traces_compressed += 1

        # Original size estimate (JSON-like)
        original_data = json.dumps({
            "trace_id": trace.trace_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "service": s.service_name,
                    "operation": s.operation_name,
                    "start_us": s.start_time_us,
                    "duration_us": s.duration_us,
                    "status": int(s.status),
                }
                for s in trace.spans
            ]
        }).encode()
        original_size = len(original_data)

        # Find or create template
        template = self._find_template(trace)
        if template:
            self.template_hits += 1
            template.frequency += 1
        else:
            self.template_misses += 1
            template = self._learn_template(trace)
            self.templates[template.template_id] = template

        # Compute deviations
        deviations = self._compute_deviations(trace, template)

        # Delta-encode timing data
        timing_data = self._delta_encode_timestamps(trace)

        # Build compressed representation
        compressed_header = json.dumps({
            "tid": trace.trace_id,
            "tpl": template.template_id,
            "dev": deviations,
        }).encode()

        # Compress the timing data
        compressed_timing = zlib.compress(timing_data, level=6)

        compressed_size = len(compressed_header) + len(compressed_timing)

        self.total_original_bytes += original_size
        self.total_compressed_bytes += compressed_size

        return CompressedTrace(
            trace_id=trace.trace_id,
            template_id=template.template_id,
            deviations=deviations,
            span_data=compressed_timing,
            original_size=original_size,
            compressed_size=compressed_size,
        )

    @property
    def overall_compression_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def template_hit_rate(self) -> float:
        total = self.template_hits + self.template_misses
        if total == 0:
            return 0.0
        return self.template_hits / total

    def stats(self) -> dict:
        return {
            "traces_compressed": self.traces_compressed,
            "original_bytes": self.total_original_bytes,
            "compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": f"{self.overall_compression_ratio:.1f}x",
            "template_hit_rate": f"{self.template_hit_rate:.1%}",
            "templates_learned": len(self.templates),
            "template_hits": self.template_hits,
            "template_misses": self.template_misses,
        }
