"""OpenTelemetry initialisation for FaultScope services.

Call ``setup_telemetry`` once at service startup.  When
``enabled=False`` (the default in local development) every call
becomes a no-op and no external dependency on a collector is required.

Example::

    from faultscope.common.telemetry import setup_telemetry, get_tracer

    setup_telemetry(
        service_name="faultscope-inference",
        enabled=True,
        endpoint="http://otel-collector:4317",
    )
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("predict"):
        ...
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Module-level flag so ``setup_telemetry`` is idempotent.
_telemetry_configured: bool = False


def setup_telemetry(
    service_name: str,
    enabled: bool,
    endpoint: str | None,
) -> None:
    """Initialise the OpenTelemetry SDK.

    When ``enabled`` is ``False`` the function returns immediately,
    leaving the no-op global tracer provider in place.  This makes it
    safe to call ``get_tracer`` in all environments without
    conditionals scattered across the code.

    The function is idempotent: subsequent calls after the first
    successful configuration are silently ignored.

    Parameters
    ----------
    service_name:
        Logical name of the service, e.g. ``"faultscope-inference"``.
        Attached to every span as the ``service.name`` resource
        attribute.
    enabled:
        When ``True``, a real ``TracerProvider`` is configured and
        spans are exported.  When ``False``, a no-op provider is used.
    endpoint:
        OTLP gRPC endpoint for the collector, e.g.
        ``"http://localhost:4317"``.  When ``None`` and ``enabled`` is
        ``True``, spans are printed to stdout via the
        ``ConsoleSpanExporter`` (useful for smoke-testing).
    """
    global _telemetry_configured  # noqa: PLW0603

    if not enabled:
        _log.info(
            "telemetry_disabled",
            service_name=service_name,
        )
        return

    if _telemetry_configured:
        _log.debug(
            "telemetry_already_configured",
            service_name=service_name,
        )
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint is not None:
        # Import lazily so the grpc exporter is only required when
        # telemetry is actually enabled.
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: E501
                OTLPSpanExporter,
            )

            exporter: OTLPSpanExporter | ConsoleSpanExporter = (
                OTLPSpanExporter(endpoint=endpoint)
            )
            _log.info(
                "telemetry_otlp_exporter_configured",
                service_name=service_name,
                endpoint=endpoint,
            )
        except ImportError as exc:
            _log.error(
                "telemetry_otlp_exporter_unavailable",
                service_name=service_name,
                endpoint=endpoint,
                error=str(exc),
            )
            raise
    else:
        exporter = ConsoleSpanExporter()
        _log.warning(
            "telemetry_using_console_exporter",
            service_name=service_name,
            reason="No OTLP endpoint provided; spans sent to stdout.",
        )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _telemetry_configured = True
    _log.info(
        "telemetry_configured",
        service_name=service_name,
    )


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for the given instrumentation scope.

    Safe to call before ``setup_telemetry``; in that case the returned
    tracer is a no-op and produces no spans.

    Parameters
    ----------
    name:
        Instrumentation scope identifier, typically ``__name__``.

    Returns
    -------
    opentelemetry.trace.Tracer
        A tracer bound to ``name`` from the global provider.
    """
    return trace.get_tracer(name)
