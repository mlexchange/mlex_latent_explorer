import os
from unittest.mock import MagicMock, patch

import pytest


# Common fixtures for MLflow testing
@pytest.fixture
def mock_mlflow_client():
    """Mock MlflowClient class"""
    with patch("src.utils.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_os_makedirs():
    """Mock os.makedirs to avoid file system errors"""
    with patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def mlflow_test_client(mock_mlflow_client, mock_os_makedirs):
    """Create a MLflowClient instance with mocked dependencies"""
    with patch("mlflow.set_tracking_uri"):  # Avoid actually setting tracking URI
        from src.utils.mlflow_utils import MLflowClient

        client = MLflowClient(
            tracking_uri="http://mock-mlflow:5000",
            username="test-user",
            password="test-password",
            cache_dir="/tmp/test_mlflow_cache",
        )
        return client


# Common fixtures for Redis testing
@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client"""
    return MagicMock()


# Updated for live_mode callbacks instead of execute
@pytest.fixture
def mock_redis_store():
    """Mock the RedisModelStore"""
    with patch("src.callbacks.live_mode.redis_model_store") as mock_store:
        yield mock_store


@pytest.fixture
def redis_test_store(mock_redis_client):
    """Create a RedisModelStore with a mock Redis client"""
    # Patch redis.Redis to return our mock
    with patch("redis.Redis", return_value=mock_redis_client):
        from src.arroyo_reduction.redis_model_store import RedisModelStore

        # Create the store - this will use our patched Redis
        store = RedisModelStore(host="localhost", port=6379)

        # Make sure our mock was used
        assert store.redis_client is mock_redis_client

        # Store the mock for tests to use
        store._mock_client = mock_redis_client

        yield store


# Common fixtures for reducer testing
@pytest.fixture
def mock_event():
    """Create a mock event with image data"""
    import numpy as np

    event = MagicMock()
    # Create test image array with proper shape and dtype
    event.image.array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return event


@pytest.fixture
def redis_mlflow_mocks():
    """Set up and start Redis and MLflow mocks"""
    # Create the patches
    redis_mock_patch = patch("src.arroyo_reduction.redis_model_store.RedisModelStore")
    mlflow_client_mock_patch = patch("src.utils.mlflow_utils.MLflowClient")

    # Start all the patches
    redis_mock = redis_mock_patch.start()
    mlflow_client_mock = mlflow_client_mock_patch.start()

    # Configure the Redis mock
    mock_store = MagicMock()
    redis_mock.return_value = mock_store
    mock_store.get_autoencoder_model.return_value = "test_autoencoder"
    mock_store.get_dimred_model.return_value = "test_dimred"

    # Configure the MLFlow mock
    mock_mlflow_client = MagicMock()
    mlflow_client_mock.return_value = mock_mlflow_client

    # Configure models to return test data
    import numpy as np

    latent_features = np.random.rand(1, 64).astype(np.float32)
    umap_coords = np.random.rand(1, 2).astype(np.float32)

    mock_autoencoder = MagicMock()
    mock_dimred = MagicMock()

    mock_autoencoder.predict.return_value = {"latent_features": latent_features}
    mock_dimred.predict.return_value = {"umap_coords": umap_coords}

    # Configure the load_model method to return appropriate models
    mock_mlflow_client.load_model.side_effect = lambda model_name: (
        mock_autoencoder if model_name == "test_autoencoder" else mock_dimred
    )

    # Store all mocks and test data in a dict
    mocks = {
        "redis": redis_mock,
        "store": mock_store,
        "mlflow_client": mock_mlflow_client,
        "autoencoder": mock_autoencoder,
        "dimred": mock_dimred,
        "latent_features": latent_features,
        "umap_coords": umap_coords,
    }

    yield mocks

    # Stop all patches after the test
    redis_mock_patch.stop()
    mlflow_client_mock_patch.stop()


# Common fixtures for model callback testing - updated for live_mode
@pytest.fixture
def mock_logger():
    """Mock the logger"""
    with patch("src.callbacks.live_mode.logger") as mock_logger:
        yield mock_logger


# Add a specific mock for the live_mode MLflow client
@pytest.fixture
def mock_live_mode_mlflow_client():
    """Mock MLflowClient for live_mode callbacks"""
    with patch("src.callbacks.live_mode.mlflow_client") as mock_client:
        # Configure check_model_compatibility for testing
        mock_client.check_model_compatibility.side_effect = (
            lambda auto, dimred: auto and dimred and auto != "incompatible"
        )
        yield mock_client
