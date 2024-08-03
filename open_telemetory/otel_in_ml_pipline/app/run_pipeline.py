import random
import time

from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# Setup Tracer
resource = Resource.create({"service.name": "my-ml-pipeline"})
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider=tracer_provider)
# Add SpanProcessor to send traces to Console, OtelCollector, and Jaeger.
tracer_provider.add_span_processor(
    span_processor=SimpleSpanProcessor(span_exporter=ConsoleSpanExporter())
)
tracer_provider.add_span_processor(
    span_processor=BatchSpanProcessor(
        span_exporter=OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
    )
)
tracer = trace.get_tracer_provider().get_tracer(__name__)


class Pipeline:
    def __init__(self):
        pass

    def run(self):
        while True:
            try:
                with tracer.start_as_current_span("process_single_frame") as root_span:
                    with trace.use_span(root_span, end_on_exit=True):
                        self.process_single_frame()
            except KeyboardInterrupt:
                break
        logger.info("Stop pipeline ...")

    def process_single_frame(self):
        # Model 1
        logger.info("Run Model 1")
        with tracer.start_as_current_span("model_1"):
            time.sleep(0.10 + random.uniform(-0.01, 0.01))

        # Model 2
        logger.info("Run Model 2")
        with tracer.start_as_current_span("model_2"):
            time.sleep(0.20 + random.uniform(-0.01, 0.01))

        # Model 3
        logger.info("Run Model 3")
        with tracer.start_as_current_span("model_3"):
            time.sleep(0.30 + random.uniform(-0.01, 0.01))


def main():
    pipeline = Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
