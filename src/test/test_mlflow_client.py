import hashlib
import os
from unittest.mock import MagicMock, call, patch

import mlflow
import pytest

from src.test.test_utils import mlflow_test_client, mock_mlflow_client, mock_os_makedirs
from mlex_utils.mlflow_utils.mlflow_model_client import MLflowModelClient


class TestMLflowClient:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset MLflowModelClient._model_cache before and after each test"""
        # Save original cache
        original_cache = MLflowModelClient._model_cache.copy()

        # Clear cache before test
        MLflowModelClient._model_cache = {}

        yield

        # Restore original cache after test
        MLflowModelClient._model_cache = original_cache

    def test_init(self, mlflow_test_client, mock_os_makedirs):
        """Test initialization of MLflowModelClient"""
        client = mlflow_test_client
        # Verify environment variables were set
        assert os.environ["MLFLOW_TRACKING_USERNAME"] == "test-user"
        assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "test-password"

        # Verify cache directory creation was attempted
        mock_os_makedirs.assert_called_with(client.cache_dir, exist_ok=True)

    def test_check_mlflow_ready_success(self, mlflow_test_client, mock_mlflow_client):
        """Test check_mlflow_ready when MLflow is reachable"""
        client = mlflow_test_client
        # Configure the mock to return a result
        mock_mlflow_client.search_experiments.return_value = []

        # Call the method
        result = client.check_mlflow_ready()

        # Verify the result is True
        assert result is True
        mock_mlflow_client.search_experiments.assert_called_once_with(max_results=1)

    def test_check_mlflow_ready_failure(self, mlflow_test_client, mock_mlflow_client):
        """Test check_mlflow_ready when MLflow is not reachable"""
        client = mlflow_test_client
        # Configure the mock to raise an exception
        mock_mlflow_client.search_experiments.side_effect = Exception(
            "Connection error"
        )

        # Call the method
        result = client.check_mlflow_ready()

        # Verify the result is False
        assert result is False
        mock_mlflow_client.search_experiments.assert_called_once_with(max_results=1)

    def test_get_mlflow_params(self, mlflow_test_client, mock_mlflow_client):
        """Test retrieving MLflow model parameters"""
        client = mlflow_test_client
        # Configure mock for get_model_version
        mock_model_version = MagicMock()
        mock_model_version.run_id = "run-123"
        mock_mlflow_client.get_model_version.return_value = mock_model_version

        # Configure mock for get_run
        mock_run = MagicMock()
        mock_run.data.params = {"param1": "value1", "param2": "value2"}
        mock_mlflow_client.get_run.return_value = mock_run

        result = client.get_mlflow_params("test-model")

        # Verify get_model_version was called with the right parameters
        mock_mlflow_client.get_model_version.assert_called_once_with(
            name="test-model", version="1"
        )

        # Verify get_run was called with the right run ID
        mock_mlflow_client.get_run.assert_called_once_with("run-123")

        # Verify the result contains the expected parameters
        assert result == {"param1": "value1", "param2": "value2"}

    def test_get_mlflow_models(self, mlflow_test_client, mock_mlflow_client):
        """Test retrieving MLflow models"""
        client = mlflow_test_client
        # Create mock model versions
        mock_version1 = MagicMock()
        mock_version1.name = "model1"
        mock_version1.version = "1"
        mock_version1.run_id = "run1"

        mock_version2 = MagicMock()
        mock_version2.name = "model2"
        mock_version2.version = "2"
        mock_version2.run_id = "run2"

        # Configure search_model_versions to return our mock versions
        mock_mlflow_client.search_model_versions.return_value = [
            mock_version1,
            mock_version2,
        ]

        # Configure runs with tags
        mock_run1 = MagicMock()
        mock_run1.data.tags = {"exp_type": "dev"}

        mock_run2 = MagicMock()
        mock_run2.data.tags = {"exp_type": "test"}

        # Configure get_run to return our mock runs
        mock_mlflow_client.get_run.side_effect = [mock_run1, mock_run2]

        # Mock the get_flow_run_name and get_flow_run_parent_id functions
        with (
            patch(
                "src.utils.mlflow_utils.get_flow_run_name", return_value="Flow Run 1"
            ),
            patch(
                "src.utils.mlflow_utils.get_flow_run_parent_id",
                return_value="parent-id",
            ),
        ):

            result = client.get_mlflow_models()

        # Verify search_model_versions was called
        mock_mlflow_client.search_model_versions.assert_called_once()

        # Verify the result has the expected structure
        assert len(result) == 2
        assert result[0]["label"] == "Flow Run 1"
        assert result[0]["value"] == "model1"
        assert result[1]["label"] == "Flow Run 1"
        assert result[1]["value"] == "model2"

    def test_get_mlflow_models_with_livemode(
        self, mlflow_test_client, mock_mlflow_client
    ):
        """Test retrieving MLflow models with livemode=True"""
        client = mlflow_test_client
        # Create mock model versions
        mock_version1 = MagicMock()
        mock_version1.name = "model1"
        mock_version1.version = "1"
        mock_version1.run_id = "run1"

        mock_version2 = MagicMock()
        mock_version2.name = "model2"
        mock_version2.version = "2"
        mock_version2.run_id = "run2"

        # Configure search_model_versions to return our mock versions
        mock_mlflow_client.search_model_versions.return_value = [
            mock_version1,
            mock_version2,
        ]

        # Configure runs with tags
        mock_run1 = MagicMock()
        mock_run1.data.tags = {"exp_type": "live_mode"}

        mock_run2 = MagicMock()
        mock_run2.data.tags = {"exp_type": "dev"}

        # Configure get_run to return our mock runs
        mock_mlflow_client.get_run.side_effect = [mock_run1, mock_run2]

        result = client.get_mlflow_models(livemode=True)

        # Verify search_model_versions was called
        mock_mlflow_client.search_model_versions.assert_called_once()

        # Verify the result contains only models with exp_type="live_mode"
        assert len(result) == 1
        assert result[0]["label"] == "model1"
        assert result[0]["value"] == "model1"

    def test_get_mlflow_models_with_model_type(
        self, mlflow_test_client, mock_mlflow_client
    ):
        """Test retrieving MLflow models with model_type filter"""
        client = mlflow_test_client
        # Create mock model versions
        mock_version1 = MagicMock()
        mock_version1.name = "model1"
        mock_version1.version = "1"
        mock_version1.run_id = "run1"

        mock_version2 = MagicMock()
        mock_version2.name = "model2"
        mock_version2.version = "2"
        mock_version2.run_id = "run2"

        # Configure search_model_versions to return our mock versions
        mock_mlflow_client.search_model_versions.return_value = [
            mock_version1,
            mock_version2,
        ]

        # Configure runs with tags
        mock_run1 = MagicMock()
        mock_run1.data.tags = {"exp_type": "dev", "model_type": "autoencoder"}

        mock_run2 = MagicMock()
        mock_run2.data.tags = {"exp_type": "dev", "model_type": "dimension_reduction"}

        # Configure get_run to return our mock runs
        mock_mlflow_client.get_run.side_effect = [mock_run1, mock_run2]

        # Mock the get_flow_run_name and get_flow_run_parent_id functions
        with (
            patch(
                "src.utils.mlflow_utils.get_flow_run_parent_id",
                return_value="parent-id",
            ),
            patch(
                "src.utils.mlflow_utils.get_flow_run_name", return_value="Flow Run 1"
            ),
        ):

            result = client.get_mlflow_models(model_type="autoencoder")

        # Verify the result contains only models with model_type "autoencoder"
        assert len(result) == 1
        assert result[0]["label"] == "Flow Run 1"
        assert result[0]["value"] == "model1"

    def test_get_cache_path(self, mlflow_test_client):
        """Test the _get_cache_path method"""
        client = mlflow_test_client
        # Test with just model name
        cache_path = client._get_cache_path("test-model")
        expected_hash = hashlib.md5("test-model".encode()).hexdigest()
        assert cache_path == f"/tmp/test_mlflow_cache/test-model_{expected_hash}"

        # Test with version
        cache_path = client._get_cache_path("test-model", 2)
        assert cache_path == "/tmp/test_mlflow_cache/test-model_v2"

    def test_load_model_from_memory_cache(self, mlflow_test_client):
        """Test loading a model from memory cache"""
        client = mlflow_test_client
        # Set up memory cache
        mock_model = MagicMock(name="memory_model")
        MLflowModelClient._model_cache = {"test-model": mock_model}

        # Load model
        result = client.load_model("test-model")

        # Verify result is from cache
        assert result is mock_model

    def test_load_model_from_disk_cache(self, mlflow_test_client, mock_mlflow_client):
        """Test loading a model from disk cache"""
        client = mlflow_test_client
        # Setup mocks
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow_client.search_model_versions.return_value = [mock_version]

        # Setup mock model
        mock_model = MagicMock(name="disk_cache_model")

        # Test disk cache path
        with (
            patch("os.path.exists", return_value=True),
            patch("mlflow.pyfunc.load_model", return_value=mock_model),
        ):

            # Load model
            result = client.load_model("test-model")

            # Verify result is the mock model
            assert result is mock_model

    def test_load_model_download_artifacts(
        self, mlflow_test_client, mock_mlflow_client
    ):
        """Test loading a model by downloading artifacts"""
        client = mlflow_test_client
        # Setup mocks
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow_client.search_model_versions.return_value = [mock_version]

        # Setup mock model
        mock_model = MagicMock(name="download_model")
        download_path = "/tmp/test_mlflow_cache/downloaded_model"

        # Test artifact download path
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
            patch("mlflow.artifacts.download_artifacts", return_value=download_path),
            patch("mlflow.pyfunc.load_model", return_value=mock_model),
        ):

            # Load model
            result = client.load_model("test-model")

            # Verify result is the mock model
            assert result is mock_model

    def test_load_model_fallback(self, mlflow_test_client, mock_mlflow_client):
        """Test load_model fallback when download fails"""
        client = mlflow_test_client
        # Setup mocks
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow_client.search_model_versions.return_value = [mock_version]

        # Setup mock model
        mock_model = MagicMock(name="fallback_model")

        # Test fallback path
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
            patch(
                "mlflow.artifacts.download_artifacts",
                side_effect=Exception("Download error"),
            ),
            patch("mlflow.pyfunc.load_model", return_value=mock_model),
        ):

            # Load model
            result = client.load_model("test-model")

            # Verify result is the mock model
            assert result is mock_model

    def test_load_model_no_versions(self, mlflow_test_client, mock_mlflow_client):
        """Test loading a model when no versions are found"""
        client = mlflow_test_client
        # Configure client to return no versions
        mock_mlflow_client.search_model_versions.return_value = []

        # Test no versions path - we don't need to patch mlflow.pyfunc.load_model
        # because it should never be called
        result = client.load_model("test-model")

        # Verify result is None
        assert result is None

    def test_load_model_error(self, mlflow_test_client, mock_mlflow_client):
        """Test error handling in load_model"""
        client = mlflow_test_client
        # Configure client to raise an exception
        mock_mlflow_client.search_model_versions.side_effect = Exception("Test error")

        # Test error path - we don't need to patch mlflow.pyfunc.load_model
        # because it should never be called
        result = client.load_model("test-model")

        # Verify result is None
        assert result is None

    def test_clear_memory_cache(self):
        """Test clearing the memory cache"""
        # Set up memory cache
        MLflowModelClient._model_cache = {"test-model": MagicMock()}

        # Clear memory cache
        MLflowModelClient.clear_memory_cache()

        # Verify memory cache is empty
        assert len(MLflowModelClient._model_cache) == 0

    def test_clear_disk_cache(self, mlflow_test_client):
        """Test clearing the disk cache"""
        client = mlflow_test_client
        # Mock shutil.rmtree and os.makedirs
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("os.makedirs") as mock_makedirs,
        ):

            # Clear disk cache
            client.clear_disk_cache()

            # Verify rmtree was called with the cache directory
            mock_rmtree.assert_called_once_with(client.cache_dir)

            # Verify makedirs was called with the cache directory
            mock_makedirs.assert_called_once_with(client.cache_dir, exist_ok=True)
