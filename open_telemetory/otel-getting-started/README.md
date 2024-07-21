# OpenTelemetry in Python - Getting Started

Source: https://opentelemetry.io/docs/languages/python/

## How to use?

```bash
make run-collector
```

```bash
poetry install
poetry shell

export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
opentelemetry-instrument --logs_exporter otlp flask run -p 8080
```
