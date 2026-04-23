"""
Pytest configuration for the entire project.

This file is loaded by pytest before any test collection or module imports,
making it the ideal place to set environment variables and global mocks.
"""
import os

# Set TESTING environment variable before any modules are imported
# This prevents Redis connections during test runs
os.environ["TESTING"] = "1"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6666"

# Disable OpenTelemetry/Prefect telemetry during testing
# This prevents background threads from trying to export traces
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"
