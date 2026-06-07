from __future__ import annotations

import atexit
import logging
import os
import time
from contextlib import contextmanager
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "on"}


class PromptTelemetry:
    """Optional OpenTelemetry bridge for PromptOrchestrator."""

    def __init__(self) -> None:
        self._initialized = False
        self._enabled = False
        self._init_error: str | None = None

        self._tracer: Any = None
        self._meter: Any = None
        self._tracer_provider: Any = None
        self._meter_provider: Any = None
        self._log_provider: Any = None
        self._otlp_logger: logging.Logger | None = None

        self._build_requests: Any = None
        self._errors: Any = None
        self._latency_ms: Any = None
        self._prompt_tokens: Any = None
        self._prompt_chars: Any = None
        self._rag_chunks: Any = None
        self._warnings_count: Any = None
        self._safety_events: Any = None
        self._summary_calls: Any = None
        self._summary_latency_ms: Any = None

    def initialize(self, service_name: str = "prompt-orchestrator") -> None:
        if self._initialized:
            return

        self._initialized = True
        enabled_raw = os.getenv("ENABLE_OTEL", "false").strip().lower()
        if enabled_raw not in TRUE_VALUES:
            return

        try:
            from opentelemetry import metrics, trace
            from opentelemetry._logs import set_logger_provider
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._init_error = str(exc)
            return

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        grpc_endpoint = endpoint.replace("http://", "").replace("https://", "")
        insecure = not endpoint.startswith("https://")
        namespace = os.getenv("OTEL_SERVICE_NAMESPACE", "prompt-stack")
        environment = os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "dev")
        version = os.getenv("OTEL_SERVICE_VERSION", "unknown")

        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
                "service.namespace": namespace,
                "service.version": version,
                "deployment.environment": environment,
            }
        )

        current_tracer_provider = trace.get_tracer_provider()
        if isinstance(current_tracer_provider, TracerProvider):
            self._tracer_provider = current_tracer_provider
        else:
            self._tracer_provider = TracerProvider(resource=resource)
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=grpc_endpoint, insecure=insecure))
            )
            trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer("prompt-orchestrator")

        current_meter_provider = metrics.get_meter_provider()
        if isinstance(current_meter_provider, MeterProvider):
            self._meter_provider = current_meter_provider
        else:
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=grpc_endpoint, insecure=insecure)
            )
            self._meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(self._meter_provider)
        self._meter = metrics.get_meter("prompt-orchestrator")

        self._build_requests = self._meter.create_counter("prompt_build_requests_total")
        self._errors = self._meter.create_counter("prompt_errors_total")
        self._latency_ms = self._meter.create_histogram("prompt_build_latency_ms", unit="ms")
        self._prompt_tokens = self._meter.create_histogram("prompt_total_tokens")
        self._prompt_chars = self._meter.create_histogram("prompt_total_chars")
        self._rag_chunks = self._meter.create_histogram("prompt_rag_chunks_count")
        self._warnings_count = self._meter.create_histogram("prompt_warnings_count")
        self._safety_events = self._meter.create_counter("prompt_safety_events_total")
        self._summary_calls = self._meter.create_counter("prompt_summary_calls_total")
        self._summary_latency_ms = self._meter.create_histogram("prompt_summary_latency_ms", unit="ms")

        from opentelemetry._logs import get_logger_provider

        current_log_provider = get_logger_provider()
        if isinstance(current_log_provider, LoggerProvider):
            self._log_provider = current_log_provider
        else:
            self._log_provider = LoggerProvider(resource=resource)
            self._log_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=grpc_endpoint, insecure=insecure))
            )
            set_logger_provider(self._log_provider)

        self._otlp_logger = logging.getLogger("prompt-orchestrator.otel")
        self._otlp_logger.setLevel(logging.INFO)
        self._otlp_logger.propagate = False
        self._otlp_logger.handlers.clear()
        self._otlp_logger.addHandler(LoggingHandler(level=logging.INFO, logger_provider=self._log_provider))

        self._enabled = True
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        if not self._initialized:
            return
        if self._meter_provider is not None:
            try:
                self._meter_provider.shutdown()
            except Exception:
                pass
        if self._tracer_provider is not None:
            try:
                self._tracer_provider.shutdown()
            except Exception:
                pass
        if self._log_provider is not None:
            try:
                self._log_provider.shutdown()
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None):
        if not self._enabled or self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span

    def record_build(
        self,
        *,
        duration_ms: float,
        total_tokens: int,
        total_chars: int,
        rag_chunks: int,
        warnings_count: int,
        safety_severity: str,
        status: str,
    ) -> None:
        if not self._enabled:
            return
        attrs = {"operation": "build_for_request", "status": status}
        self._build_requests.add(1, attrs)
        self._latency_ms.record(duration_ms, attrs)
        self._prompt_tokens.record(max(total_tokens, 0), attrs)
        self._prompt_chars.record(max(total_chars, 0), attrs)
        self._rag_chunks.record(max(rag_chunks, 0), attrs)
        self._warnings_count.record(max(warnings_count, 0), attrs)
        self._safety_events.add(1, {"severity": safety_severity or "unknown", "status": status})

    def record_summary_call(self, *, duration_ms: float, provider: str, status: str) -> None:
        if not self._enabled:
            return
        attrs = {"operation": "summary", "provider": provider, "status": status}
        self._summary_calls.add(1, attrs)
        self._summary_latency_ms.record(duration_ms, attrs)

    def record_error(self, operation: str, error_type: str) -> None:
        if not self._enabled:
            return
        self._errors.add(1, {"operation": operation, "error.type": error_type})
        self.emit_log(level_name="ERROR", message=f"prompt.error operation={operation} error_type={error_type}")

    def emit_log(self, *, level_name: str, message: str) -> None:
        if not self._enabled or self._otlp_logger is None:
            return

        level = logging.INFO
        normalized = level_name.strip().upper()
        if normalized == "DEBUG":
            level = logging.DEBUG
        elif normalized in {"WARN", "WARNING"}:
            level = logging.WARNING
        elif normalized == "ERROR":
            level = logging.ERROR
        elif normalized == "CRITICAL":
            level = logging.CRITICAL
        self._otlp_logger.log(level, message)


telemetry = PromptTelemetry()


def init_telemetry(service_name: str = "prompt-orchestrator") -> None:
    telemetry.initialize(service_name=service_name)


def shutdown_telemetry() -> None:
    telemetry.shutdown()


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0
