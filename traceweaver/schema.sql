-- TraceWeaver ClickHouse Schema
-- Optimized for trace storage with aggressive compression

CREATE DATABASE IF NOT EXISTS traceweaver;

-- Main traces table
CREATE TABLE IF NOT EXISTS traceweaver.traces (
    trace_id String,
    timestamp DateTime DEFAULT now(),
    span_count UInt32,
    duration_us UInt64,
    root_service String,
    root_operation String,
    is_anomaly UInt8 DEFAULT 0,
    anomaly_score Float64 DEFAULT 0.0,
    compression_ratio Float32 DEFAULT 0.0,
    template_id String DEFAULT ''
) ENGINE = MergeTree()
ORDER BY (timestamp, trace_id)
PARTITION BY toYYYYMM(timestamp)
SETTINGS index_granularity = 8192;

-- Spans table (child of traces)
CREATE TABLE IF NOT EXISTS traceweaver.spans (
    trace_id String,
    span_id String,
    parent_id String DEFAULT '',
    timestamp DateTime DEFAULT now(),
    service_name LowCardinality(String),
    operation_name LowCardinality(String),
    start_time_us UInt64,
    duration_us UInt64,
    status Enum8('UNSET' = 0, 'OK' = 1, 'ERROR' = 2)
) ENGINE = MergeTree()
ORDER BY (timestamp, service_name, trace_id)
PARTITION BY toYYYYMM(timestamp)
SETTINGS index_granularity = 8192;

-- Compression stats table
CREATE TABLE IF NOT EXISTS traceweaver.compression_stats (
    timestamp DateTime DEFAULT now(),
    traces_processed UInt64,
    original_bytes UInt64,
    compressed_bytes UInt64,
    compression_ratio Float32,
    template_hit_rate Float32,
    templates_count UInt32
) ENGINE = MergeTree()
ORDER BY timestamp
PARTITION BY toYYYYMM(timestamp);

-- Materialized view for real-time service latency percentiles
CREATE MATERIALIZED VIEW IF NOT EXISTS traceweaver.service_latency_mv
ENGINE = AggregatingMergeTree()
ORDER BY (time_bucket, service_name)
AS SELECT
    toStartOfMinute(timestamp) AS time_bucket,
    service_name,
    quantileState(0.5)(duration_us) AS p50_latency,
    quantileState(0.95)(duration_us) AS p95_latency,
    quantileState(0.99)(duration_us) AS p99_latency,
    countState() AS request_count,
    countIfState(status = 'ERROR') AS error_count
FROM traceweaver.spans
GROUP BY time_bucket, service_name;
