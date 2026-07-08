"""Streaming anomaly detection using t-digest quantile estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .trace import Trace


@dataclass
class Centroid:
    """A single centroid in the t-digest."""
    mean: float
    count: int


class TDigest:
    """
    T-Digest: streaming quantile estimation with bounded memory.

    Based on Ted Dunning's t-digest algorithm.
    Maintains a compressed set of centroids that approximate the
    distribution of a data stream. Supports merge operations.
    """

    def __init__(self, compression: float = 100.0):
        self.compression = compression
        self.centroids: list[Centroid] = []
        self.count = 0
        self._unmerged: list[float] = []
        self._max_unmerged = 500

    def update(self, value: float) -> None:
        """Add a single value to the digest."""
        self._unmerged.append(value)
        if len(self._unmerged) >= self._max_unmerged:
            self._merge_buffer()

    def _merge_buffer(self) -> None:
        """Merge buffered values into centroids."""
        if not self._unmerged:
            return

        values = sorted(self._unmerged)
        self._unmerged.clear()

        # Merge with existing centroids
        all_points = [(v, 1) for v in values]
        for c in self.centroids:
            all_points.append((c.mean, c.count))
        all_points.sort(key=lambda x: x[0])

        # Rebuild centroids with compression limit
        new_centroids: list[Centroid] = []
        total = sum(w for _, w in all_points)
        cumulative = 0.0

        current_mean = 0.0
        current_count = 0

        for value, weight in all_points:
            if current_count == 0:
                current_mean = value
                current_count = weight
                cumulative += weight
                continue

            # Check if we can merge into current centroid
            proposed_count = current_count + weight
            q = (cumulative + proposed_count / 2) / total
            max_count = self._scale(q) * total

            if proposed_count <= max_count:
                # Merge
                current_mean = (current_mean * current_count + value * weight) / proposed_count
                current_count = proposed_count
            else:
                # Flush current centroid
                new_centroids.append(Centroid(mean=current_mean, count=current_count))
                cumulative += current_count
                current_mean = value
                current_count = weight

        if current_count > 0:
            new_centroids.append(Centroid(mean=current_mean, count=current_count))

        self.centroids = new_centroids
        self.count = total

    def _scale(self, q: float) -> float:
        """Scale function for t-digest compression."""
        return self.compression / (2 * math.pi) * math.asin(2 * q - 1)

    def quantile(self, q: float) -> float:
        """Estimate the q-th quantile (0 <= q <= 1)."""
        self._merge_buffer()

        if not self.centroids:
            raise ValueError("No data in digest")

        if q <= 0:
            return self.centroids[0].mean
        if q >= 1:
            return self.centroids[-1].mean

        target = q * self.count
        cumulative = 0.0

        for i, c in enumerate(self.centroids):
            if cumulative + c.count >= target:
                if i == 0:
                    return c.mean
                # Linear interpolation between centroids
                prev = self.centroids[i - 1]
                fraction = (target - cumulative) / c.count
                return prev.mean + fraction * (c.mean - prev.mean)
            cumulative += c.count

        return self.centroids[-1].mean

    @property
    def mean(self) -> float:
        self._merge_buffer()
        if not self.centroids:
            return 0.0
        total_weight = sum(c.count for c in self.centroids)
        return sum(c.mean * c.count for c in self.centroids) / total_weight

    @property
    def std(self) -> float:
        self._merge_buffer()
        if len(self.centroids) < 2:
            return 0.0
        m = self.mean
        total = sum(c.count for c in self.centroids)
        variance = sum(c.count * (c.mean - m) ** 2 for c in self.centroids) / total
        return math.sqrt(variance)

    def merge(self, other: TDigest) -> TDigest:
        """Merge two t-digests."""
        result = TDigest(compression=self.compression)
        result._unmerged = []
        for c in self.centroids:
            result._unmerged.extend([c.mean] * c.count)
        for c in other.centroids:
            result._unmerged.extend([c.mean] * c.count)
        result._merge_buffer()
        return result


@dataclass
class AnomalyScore:
    """Result of anomaly detection on a trace."""
    trace_id: str
    is_anomaly: bool
    latency_z_score: float
    span_count_z_score: float
    error_flag: bool
    combined_score: float
    reason: str = ""


class AnomalyDetector:
    """
    Streaming anomaly detector for distributed traces.

    Maintains per-service latency distributions using t-digest,
    and flags traces that deviate significantly from normal patterns.
    """

    def __init__(self, z_threshold: float = 2.5, window_size: int = 10000):
        self.z_threshold = z_threshold
        self.window_size = window_size

        # Per-service latency distributions
        self.latency_digests: dict[str, TDigest] = {}
        # Global span count distribution
        self.span_count_digest = TDigest(compression=50)
        # Per-service error rate tracking
        self.error_counts: dict[str, int] = {}
        self.total_counts: dict[str, int] = {}

    def _get_digest(self, service: str) -> TDigest:
        if service not in self.latency_digests:
            self.latency_digests[service] = TDigest(compression=50)
        return self.latency_digests[service]

    def update(self, trace: Trace) -> None:
        """Update distributions with a new trace (no anomaly scoring)."""
        root = trace.root_span
        if root:
            digest = self._get_digest(root.service_name)
            digest.update(root.duration_us)

        self.span_count_digest.update(len(trace.spans))

        for span in trace.spans:
            self.total_counts[span.service_name] = (
                self.total_counts.get(span.service_name, 0) + 1
            )
            if span.status == 2:  # ERROR
                self.error_counts[span.service_name] = (
                    self.error_counts.get(span.service_name, 0) + 1
                )

    def score(self, trace: Trace) -> AnomalyScore:
        """Score a trace for anomalies."""
        root = trace.root_span
        if not root:
            return AnomalyScore(
                trace_id=trace.trace_id,
                is_anomaly=False,
                latency_z_score=0.0,
                span_count_z_score=0.0,
                error_flag=False,
                combined_score=0.0,
                reason="no root span",
            )

        # Latency z-score
        digest = self._get_digest(root.service_name)
        latency_z = 0.0
        if digest.count > 10:
            mean = digest.mean
            std = digest.std
            if std > 0:
                latency_z = abs(root.duration_us - mean) / std

        # Span count z-score
        span_count_z = 0.0
        if self.span_count_digest.count > 10:
            sc_mean = self.span_count_digest.mean
            sc_std = self.span_count_digest.std
            if sc_std > 0:
                span_count_z = abs(len(trace.spans) - sc_mean) / sc_std

        # Error flag
        has_error = any(s.status == 2 for s in trace.spans)

        # Combined score (weighted)
        combined = 0.5 * latency_z + 0.3 * span_count_z + (0.2 if has_error else 0.0)

        is_anomaly = combined > self.z_threshold or (has_error and latency_z > 2.0 and combined > 1.5)

        reason_parts = []
        if latency_z > self.z_threshold:
            reason_parts.append(f"latency_z={latency_z:.2f}")
        if span_count_z > self.z_threshold:
            reason_parts.append(f"span_count_z={span_count_z:.2f}")
        if has_error:
            reason_parts.append("has_errors")

        return AnomalyScore(
            trace_id=trace.trace_id,
            is_anomaly=is_anomaly,
            latency_z_score=latency_z,
            span_count_z_score=span_count_z,
            error_flag=has_error,
            combined_score=combined,
            reason=", ".join(reason_parts) if reason_parts else "normal",
        )
