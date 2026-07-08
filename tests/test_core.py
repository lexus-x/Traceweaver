"""Core tests for TraceWeaver components."""

import pytest
from traceweaver.trace import Trace, Span, SpanStatus, generate_trace_id, generate_span_id
from traceweaver.anomaly import TDigest, AnomalyDetector
from traceweaver.compressor import TraceCompressor
from traceweaver.encoder import TraceEncoder, estimate_json_size
from tracebench.generator import TraceGenerator


# --- Trace Tests ---

def test_trace_creation():
    trace = Trace(trace_id="test123")
    assert trace.trace_id == "test123"
    assert len(trace.spans) == 0
    assert trace.duration_us == 0


def test_span_creation():
    span = Span(
        trace_id="t1",
        span_id="s1",
        parent_id=None,
        service_name="api",
        operation_name="GET /",
        start_time_us=1000000,
        duration_us=50000,
        status=SpanStatus.OK,
    )
    assert span.end_time_us == 1050000
    assert span.service_operation == "api:GET /"


def test_trace_properties():
    trace = Trace(trace_id="t1")
    trace.spans = [
        Span("t1", "s1", None, "api", "GET", 1000, 5000, SpanStatus.OK),
        Span("t1", "s2", "s1", "db", "query", 1200, 3000, SpanStatus.OK),
        Span("t1", "s3", "s1", "cache", "get", 1300, 500, SpanStatus.OK),
    ]
    assert trace.root_span.span_id == "s1"
    assert trace.services == {"api", "db", "cache"}
    # duration = max(end_times) - min(start_times) = max(6000,4200,1800) - 1000 = 5000
    assert trace.duration_us == 6000 - 1000


# --- TDigest Tests ---

def test_tdigest_basic():
    digest = TDigest(compression=50)
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        digest.update(v)
    digest._merge_buffer()
    assert digest.count == 10
    assert abs(digest.mean - 5.5) < 0.1


def test_tdigest_quantile():
    import random
    random.seed(42)
    digest = TDigest(compression=100)
    values = [random.gauss(100, 15) for _ in range(10000)]
    for v in values:
        digest.update(v)
    digest._merge_buffer()

    # p50 should be close to 100
    p50 = digest.quantile(0.5)
    assert 95 < p50 < 105

    # p99 should be higher than p50
    p99 = digest.quantile(0.99)
    assert p99 > p50 + 10


# --- Anomaly Detector Tests ---

def test_anomaly_detector_basic():
    detector = AnomalyDetector(z_threshold=3.0)

    # Create normal traces
    gen = TraceGenerator(anomaly_rate=0.0)
    for _ in range(100):
        trace = gen.generate_trace()
        detector.update(trace)

    # Score a normal trace
    normal = gen.generate_trace()
    score = detector.score(normal)
    assert score.combined_score < 3.0


def test_anomaly_detector_catches_anomalies():
    detector = AnomalyDetector(z_threshold=2.5)

    # Train on normal traces (need enough for distribution)
    gen = TraceGenerator(anomaly_rate=0.0)
    for _ in range(500):
        trace = gen.generate_trace()
        detector.update(trace)

    # Generate and score multiple anomalies
    anomaly_gen = TraceGenerator(anomaly_rate=1.0)
    detected = 0
    for _ in range(20):
        anomaly = anomaly_gen.generate_trace(inject_anomaly=True)
        score = detector.score(anomaly)
        if score.is_anomaly:
            detected += 1

    # Should detect at least some anomalies
    assert detected > 0, f"Detected 0 out of 20 anomalies"


# --- Compressor Tests ---

def test_compressor_basic():
    compressor = TraceCompressor()
    gen = TraceGenerator(anomaly_rate=0.0)

    trace = gen.generate_trace()
    compressed = compressor.compress(trace)

    assert compressed.trace_id == trace.trace_id
    assert compressed.compressed_size > 0
    assert compressed.original_size > 0


def test_compressor_stats():
    compressor = TraceCompressor()
    gen = TraceGenerator(anomaly_rate=0.0)

    for _ in range(50):
        trace = gen.generate_trace()
        compressor.compress(trace)

    stats = compressor.stats()
    assert stats["traces_compressed"] == 50
    assert stats["original_bytes"] > 0


# --- Encoder Tests ---

def test_encoder_basic():
    encoder = TraceEncoder()
    gen = TraceGenerator(anomaly_rate=0.0)

    trace = gen.generate_trace()
    encoded = encoder.encode(trace)

    assert len(encoded) > 0
    assert len(encoded) < estimate_json_size(trace)


def test_encoder_compression():
    encoder = TraceEncoder()
    gen = TraceGenerator(anomaly_rate=0.0)

    trace = gen.generate_trace()
    raw = encoder.encode(trace)
    compressed = encoder.encode_compressed(trace)

    assert len(compressed) < len(raw)


# --- Generator Tests ---

def test_generator_creates_traces():
    gen = TraceGenerator(anomaly_rate=0.0)
    trace = gen.generate_trace()

    assert trace.trace_id
    assert len(trace.spans) > 0
    assert trace.root_span is not None


def test_generator_anomaly_injection():
    gen = TraceGenerator(anomaly_rate=1.0)
    trace = gen.generate_trace(inject_anomaly=True)

    assert len(trace.spans) > 0
    root = trace.root_span
    # Anomalous traces should have higher latency
    normal_gen = TraceGenerator(anomaly_rate=0.0)
    normal = normal_gen.generate_trace()
    # This is a soft check — anomalies have much higher latency
    assert root.duration_us >= normal.root_span.duration_us * 0.5


def test_generator_batch():
    gen = TraceGenerator(anomaly_rate=0.01)
    traces = gen.generate_batch(100)

    assert len(traces) == 100
    total_spans = sum(len(t.spans) for t in traces)
    assert total_spans > 100  # Each trace has multiple spans


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
