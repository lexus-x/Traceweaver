"""Synthetic trace generator with realistic microservice topologies."""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass

import numpy as np

from traceweaver.trace import Span, SpanStatus, Trace, generate_span_id, generate_trace_id


@dataclass
class ServiceConfig:
    """Configuration for a single microservice."""
    name: str
    operations: list[str]
    base_latency_us: int  # microseconds
    latency_std_us: int
    error_rate: float = 0.01  # 1% error rate


@dataclass
class TopologyConfig:
    """Configuration for the service mesh topology."""
    services: list[ServiceConfig]
    call_graph: dict[str, list[str]]  # service -> [called services]
    max_depth: int = 6
    branching_factor: float = 1.5  # avg children per span


# Realistic service topologies
E_COMMERCE_TOPOLOGY = TopologyConfig(
    services=[
        ServiceConfig("api-gateway", ["POST /order", "GET /products", "POST /cart"], 5000, 2000),
        ServiceConfig("order-service", ["create_order", "validate_order", "process_payment"], 15000, 8000),
        ServiceConfig("product-service", ["get_product", "search", "update_inventory"], 8000, 3000),
        ServiceConfig("cart-service", ["add_item", "remove_item", "get_cart"], 5000, 2000),
        ServiceConfig("payment-service", ["charge", "refund", "verify"], 25000, 12000),
        ServiceConfig("inventory-service", ["check_stock", "reserve", "release"], 10000, 5000),
        ServiceConfig("user-service", ["authenticate", "get_profile", "update_preferences"], 7000, 3000),
        ServiceConfig("notification-service", ["send_email", "send_sms", "push_notification"], 8000, 4000),
        ServiceConfig("shipping-service", ["calculate_rate", "create_shipment", "track"], 12000, 6000),
        ServiceConfig("recommendation-service", ["get_recommendations", "update_model"], 30000, 15000),
    ],
    call_graph={
        "api-gateway": ["order-service", "product-service", "cart-service", "user-service"],
        "order-service": ["payment-service", "inventory-service", "notification-service", "shipping-service"],
        "product-service": ["inventory-service", "recommendation-service"],
        "cart-service": ["product-service", "user-service"],
        "payment-service": ["notification-service"],
        "inventory-service": [],
        "user-service": [],
        "notification-service": [],
        "shipping-service": [],
        "recommendation-service": ["product-service"],
    },
    max_depth=5,
    branching_factor=1.8,
)


class TraceGenerator:
    """
    Generates realistic synthetic distributed traces.

    Features:
    - Configurable service mesh topology
    - Zipf-distributed call patterns
    - Injected latency anomalies (configurable percentage)
    - Realistic timing with parent-child dependencies
    """

    def __init__(self, topology: TopologyConfig | None = None, anomaly_rate: float = 0.01):
        self.topology = topology or E_COMMERCE_TOPOLOGY
        self.anomaly_rate = anomaly_rate
        self._service_map = {s.name: s for s in self.topology.services}

    def generate_trace(self, inject_anomaly: bool | None = None) -> Trace:
        """Generate a single synthetic trace."""
        trace_id = generate_trace_id()
        trace = Trace(trace_id=trace_id)

        if inject_anomaly is None:
            inject_anomaly = random.random() < self.anomaly_rate

        # Start with the entry point (api-gateway)
        entry_service = self.topology.services[0]
        entry_op = random.choice(entry_service.operations)

        base_time = int(time.time() * 1_000_000)  # current time in microseconds

        self._generate_spans(
            trace=trace,
            service_name=entry_service.name,
            operation=entry_op,
            parent_id=None,
            start_time=base_time,
            depth=0,
            inject_anomaly=inject_anomaly,
        )

        return trace

    def _generate_spans(
        self,
        trace: Trace,
        service_name: str,
        operation: str,
        parent_id: str | None,
        start_time: int,
        depth: int,
        inject_anomaly: bool,
    ) -> None:
        """Recursively generate spans for a trace."""
        if depth >= self.topology.max_depth:
            return

        service = self._service_map.get(service_name)
        if not service:
            return

        span_id = generate_span_id()

        # Calculate latency
        if inject_anomaly and depth == 0:
            # Anomalous root span: 8-30x normal latency
            multiplier = random.uniform(8, 30)
            duration = int(service.base_latency_us * multiplier)
            status = SpanStatus.ERROR if random.random() < 0.7 else SpanStatus.OK
        else:
            duration = max(100, int(np.random.normal(
                service.base_latency_us, service.latency_std_us
            )))
            status = SpanStatus.ERROR if random.random() < service.error_rate else SpanStatus.OK

        span = Span(
            trace_id=trace.trace_id,
            span_id=span_id,
            parent_id=parent_id,
            service_name=service_name,
            operation_name=operation,
            start_time_us=start_time,
            duration_us=duration,
            status=status,
        )
        trace.spans.append(span)

        # Generate child spans
        callees = self.topology.call_graph.get(service_name, [])
        if not callees:
            return

        # Zipf-distributed number of children
        num_children = min(
            len(callees),
            max(1, int(np.random.zipf(1.5)))
        )
        # Cap at branching factor
        num_children = min(num_children, int(self.topology.branching_factor * 2))

        # Select which services to call (weighted towards more common paths)
        weights = [1.0 / (i + 1) for i in range(len(callees))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        called_services = np.random.choice(
            callees, size=min(num_children, len(callees)), replace=False, p=weights
        )

        child_start = start_time + int(duration * 0.1)  # 10% into parent span

        for called_service in called_services:
            callee_config = self._service_map.get(called_service)
            if not callee_config:
                continue

            callee_op = random.choice(callee_config.operations)

            self._generate_spans(
                trace=trace,
                service_name=called_service,
                operation=callee_op,
                parent_id=span_id,
                start_time=child_start,
                depth=depth + 1,
                inject_anomaly=inject_anomaly and random.random() < 0.3,
            )
            # Stagger child start times
            child_start += int(np.random.exponential(1000))

    def generate_batch(self, count: int, anomaly_positions: set[int] | None = None) -> list[Trace]:
        """Generate a batch of traces with optional pre-set anomaly positions."""
        traces = []
        for i in range(count):
            inject = None
            if anomaly_positions and i in anomaly_positions:
                inject = True
            traces.append(self.generate_trace(inject_anomaly=inject))
        return traces

    def generate_dataset(
        self,
        normal_count: int = 9900,
        anomaly_count: int = 100,
    ) -> tuple[list[Trace], list[Trace]]:
        """Generate a labeled dataset: (normal_traces, anomaly_traces)."""
        normal = [self.generate_trace(inject_anomaly=False) for _ in range(normal_count)]
        anomalies = [self.generate_trace(inject_anomaly=True) for _ in range(anomaly_count)]
        return normal, anomalies
