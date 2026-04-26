import os
from prometheus_client import Counter, Histogram, Gauge, push_to_gateway, CollectorRegistry

REGISTRY = CollectorRegistry()

request_count = Counter(
    "chandra_requests_total",
    "Total OCR requests",
    ["endpoint", "status"],
    registry=REGISTRY,
)

request_latency = Histogram(
    "chandra_request_latency_seconds",
    "OCR request latency",
    ["endpoint"],
    registry=REGISTRY,
)

active_instances = Gauge(
    "chandra_active_serving_instances",
    "Number of active Vast.ai serving instances",
    registry=REGISTRY,
)

sample_count = Gauge(
    "chandra_sample_count_total",
    "Total samples accumulated in HF dataset repo",
    registry=REGISTRY,
)

def push_metrics(job: str = "chandra") -> None:
    url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    push_to_gateway(url, job=job, registry=REGISTRY)
