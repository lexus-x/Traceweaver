"""Core data structures for distributed traces."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class SpanStatus(IntEnum):
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class Span:
    """A single unit of work in a distributed trace."""

    trace_id: str
    span_id: str
    parent_id: Optional[str]
    service_name: str
    operation_name: str
    start_time_us: int  # microseconds since epoch
    duration_us: int  # microseconds
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def end_time_us(self) -> int:
        return self.start_time_us + self.duration_us

    @property
    def service_operation(self) -> str:
        return f"{self.service_name}:{self.operation_name}"


@dataclass
class Trace:
    """A complete distributed trace — a DAG of spans."""

    trace_id: str
    spans: list[Span] = field(default_factory=list)

    @property
    def root_span(self) -> Optional[Span]:
        for span in self.spans:
            if span.parent_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def duration_us(self) -> int:
        if not self.spans:
            return 0
        start = min(s.start_time_us for s in self.spans)
        end = max(s.end_time_us for s in self.spans)
        return end - start

    @property
    def services(self) -> set[str]:
        return {s.service_name for s in self.spans}

    @property
    def depth(self) -> int:
        """Max depth of the trace tree."""
        if not self.spans:
            return 0
        parent_map: dict[str | None, list[str]] = {}
        for s in self.spans:
            parent_map.setdefault(s.parent_id, []).append(s.span_id)

        def _depth(node_id: str | None, memo: dict[str, int]) -> int:
            if node_id in memo:
                return memo[node_id]
            children = parent_map.get(node_id, [])
            if not children:
                memo[node_id or ""] = 0
                return 0
            d = 1 + max(_depth(c, memo) for c in children)
            memo[node_id or ""] = d
            return d

        return _depth(None, {})

    def adjacency_list(self) -> dict[str | None, list[str]]:
        """Build parent -> [children] adjacency list."""
        adj: dict[str | None, list[str]] = {}
        for s in self.spans:
            adj.setdefault(s.parent_id, []).append(s.span_id)
        return adj

    def graph_signature(self) -> str:
        """Generate a structural signature of the trace graph (ignoring timing)."""
        edges = []
        for s in self.spans:
            edges.append((s.parent_id or "root", s.span_id, s.service_operation))
        # Sort for deterministic signature
        edges.sort()
        return "|".join(f"{p}->{c}:{op}" for p, c, op in edges)


def generate_trace_id() -> str:
    return uuid.uuid4().hex


def generate_span_id() -> str:
    return uuid.uuid4().hex[:16]
