"""
Pytest configuration and fixtures for the test suite.

This module sets up mocks and patches that need to be in place
before any application modules are imported.
"""
import os
import sys
from unittest.mock import MagicMock, patch

# CRITICAL: Set environment variable at module import time,
# BEFORE any test modules or application modules are imported
os.environ["TESTING"] = "1"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6666"

import pytest


# Patch Redis at import time to prevent connection attempts during module loading
@pytest.fixture(scope="session", autouse=True)
def mock_redis_at_import():
    """
    Mock Redis connection at the session level to prevent connection
    attempts when modules are first imported.
    """
    with patch("redis.Redis") as mock_redis_class:
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        
        # Configure mock Redis client
        mock_redis_instance.get.return_value = None
        mock_redis_instance.set.return_value = True
        mock_redis_instance.pubsub.return_value = MagicMock()
        
        yield mock_redis_instance


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules between tests if needed"""
    yield
    # Cleanup can be added here if needed
