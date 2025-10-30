from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.test.test_utils import mock_event, redis_mlflow_mocks


class TestReducer:

    @pytest.fixture
    def reducer(self, redis_mlflow_mocks):
        """Create a LatentSpaceReducer with all dependencies mocked"""
        # Patch _subscribe_to_model_updates to avoid threading issues
        with patch(
            "src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates"
        ):
            # Import here after mocks are set up
            from src.arroyo_reduction.reducer import LatentSpaceReducer

            # Create the reducer with all dependencies mocked
            reducer = LatentSpaceReducer()

            # Set references to the mock models
            reducer.current_torch_model = redis_mlflow_mocks["autoencoder"]
            reducer.current_dim_reduction_model = redis_mlflow_mocks["dimred"]

            # Store test data
            reducer._test_data = {"mocks": redis_mlflow_mocks}

            return reducer

    def test_reduce(self, reducer, mock_event, redis_mlflow_mocks):
        """Test that reduce() correctly processes an image through models"""
        # Get the mocks
        mocks = redis_mlflow_mocks

        # Create real numpy arrays with proper dimensions and types
        latent_features = np.random.rand(1, 64).astype(np.float32)
        dimred_coords = np.random.rand(1, 2).astype(np.float32)  # CHANGED: umap_coords -> dimred_coords

        # Set up the mocks to return real numpy arrays
        mocks["autoencoder"].predict.return_value = {"latent_features": latent_features}
        mocks["dimred"].predict.return_value = {"coords": dimred_coords}  # CHANGED: "umap_coords" -> "coords"

        # Set the models
        reducer.current_torch_model = mocks["autoencoder"]
        reducer.current_dim_reduction_model = mocks["dimred"]

        # Mock logger to prevent logging issues
        with patch("src.arroyo_reduction.reducer.logger"):
            # Call reduce()
            result, timing_info = reducer.reduce(mock_event)

        # Verify the processing flow

        # 1. The autoencoder should be called with the correct input
        mocks["autoencoder"].predict.assert_called_once()

        # 2. The dimred model should be called with the latent features
        mocks["dimred"].predict.assert_called_once_with(latent_features)

        # 3. The result should be the actual numpy array from the mocked dimred_coords
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)  # Check expected shape of dimensionality reduction coordinates
        # Verify it's actually the same array we created
        np.testing.assert_array_equal(result, dimred_coords)  # CHANGED: umap_coords -> dimred_coords
        
        # 4. Verify timing info is returned
        assert isinstance(timing_info, dict)
        assert 'autoencoder_time' in timing_info
        assert 'dimred_time' in timing_info

    def test_reduce_during_model_loading(self, reducer, mock_event):
        """Test that reduce() returns None when models are loading"""
        # Set loading flag to True to simulate model loading
        reducer.is_loading_model = True
        reducer.loading_model_type = "autoencoder"

        # Mock logger to prevent logging issues
        with patch("src.arroyo_reduction.reducer.logger"):
            # Call reduce()
            result, timing_info = reducer.reduce(mock_event)

        # Verify result is None when models are loading
        assert result is None
        assert isinstance(timing_info, dict)
        
        # Models should not be called
        reducer.current_torch_model.predict.assert_not_called()
        reducer.current_dim_reduction_model.predict.assert_not_called()

    def test_init_loads_models_from_redis(self):
        """Test that constructor loads models from Redis with version support"""
        # Create mocks for dependencies
        with (
            patch(
                "src.arroyo_reduction.redis_model_store.RedisModelStore", autospec=True
            ) as redis_class_mock,
            patch("src.utils.mlflow_utils.MLflowClient") as mlflow_client_mock,
            patch(
                "src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates"
            ),
            patch("src.arroyo_reduction.reducer.logger"),
        ):

            # Set up mock store - return identifiers with versions
            mock_store = MagicMock()
            redis_class_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder:3"
            mock_store.get_dimred_model.return_value = "test_dimred:2"

            # Set up mock MLflowClient
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client

            # Set up mock models
            mock_autoencoder = MagicMock()
            mock_dimred = MagicMock()

            # Configure the load_model method to handle version parameter
            def load_model_side_effect(name, version=None):
                if 'autoencoder' in name:
                    return mock_autoencoder
                return mock_dimred
            
            mock_mlflow_client.load_model.side_effect = load_model_side_effect

            # Import the real class - this needs to be done AFTER setting up the mocks
            import sys
            if "src.arroyo_reduction.reducer" in sys.modules:
                del sys.modules["src.arroyo_reduction.reducer"]

            from src.arroyo_reduction.reducer import LatentSpaceReducer

            # Create the reducer
            reducer = LatentSpaceReducer()

            # Verify RedisModelStore was instantiated
            redis_class_mock.assert_called()

            # Verify models were requested
            mock_store.get_autoencoder_model.assert_called()
            mock_store.get_dimred_model.assert_called()

            # Verify MLflowClient was instantiated
            mlflow_client_mock.assert_called()

            # Verify models were loaded with parsed version
            assert mock_mlflow_client.load_model.call_count == 2
            calls = mock_mlflow_client.load_model.call_args_list
            
            # First call should be autoencoder with version='3'
            assert calls[0][0][0] == "test_autoencoder"
            assert calls[0][1]['version'] == '3'
            
            # Second call should be dimred with version='2'
            assert calls[1][0][0] == "test_dimred"
            assert calls[1][1]['version'] == '2'

            # Verify loading flags are set correctly during initialization
            assert not reducer.is_loading_model
            assert reducer.loading_model_type is None

    def test_handle_model_update(self):
        """Test handling model update notifications with version support"""
        # Set up mock models
        mock_orig_autoencoder = MagicMock()
        mock_new_autoencoder = MagicMock()
        mock_orig_dimred = MagicMock()
        mock_new_dimred = MagicMock()

        # Create the patch for MLflowClient and redis before importing anything
        with (
            patch("src.utils.mlflow_utils.MLflowClient") as mlflow_client_mock,
            patch(
                "src.arroyo_reduction.redis_model_store.RedisModelStore"
            ) as redis_mock,
            patch(
                "src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates"
            ),
            patch("src.arroyo_reduction.reducer.logger"),
        ):

            # Configure MLflowClient mock
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client

            # Configure load_model to return different models and handle version parameter
            def load_model_side_effect(name, version=None):
                model_map = {
                    "test_autoencoder": mock_orig_autoencoder,
                    "new_autoencoder": mock_new_autoencoder,
                    "test_dimred": mock_orig_dimred,
                    "new_dimred": mock_new_dimred,
                }
                return model_map.get(name, MagicMock())
            
            mock_mlflow_client.load_model.side_effect = load_model_side_effect

            # Set up mock store - return identifiers without versions for initial load
            mock_store = MagicMock()
            redis_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder"
            mock_store.get_dimred_model.return_value = "test_dimred"

            # Import the real class
            import sys
            if "src.arroyo_reduction.reducer" in sys.modules:
                del sys.modules["src.arroyo_reduction.reducer"]

            from src.arroyo_reduction.reducer import LatentSpaceReducer

            # Create the reducer - which loads the initial models
            reducer = LatentSpaceReducer()

            # Add mlflow_client attribute with load_model method
            reducer.mlflow_client = mock_mlflow_client

            # Verify initial models were loaded
            assert reducer.autoencoder_model_name == "test_autoencoder"
            assert reducer.dimred_model_name == "test_dimred"

            # Create update message with version in identifier
            autoencoder_update = {
                "model_type": "autoencoder",
                "model_name": "new_autoencoder:5",
            }

            # Handle autoencoder update
            reducer._handle_model_update(autoencoder_update)

            # Verify model name was updated with version
            assert reducer.autoencoder_model_name == "new_autoencoder:5"

            # Verify load_model was called with parsed name and version
            mock_mlflow_client.load_model.assert_any_call("new_autoencoder", version="5")

            # Create update for dimred model with version
            dimred_update = {
                "model_type": "dimred", 
                "model_name": "new_dimred:3"
            }

            # Handle dimred update
            reducer._handle_model_update(dimred_update)

            # Verify dimred model name was updated with version
            assert reducer.dimred_model_name == "new_dimred:3"

            # Verify load_model was called with parsed name and version
            mock_mlflow_client.load_model.assert_any_call("new_dimred", version="3")

            # Test with invalid update
            invalid_update = {"model_type": "unknown", "model_name": "test"}
            reducer._handle_model_update(invalid_update)

            # Verify no changes with invalid update
            assert reducer.autoencoder_model_name == "new_autoencoder:5"
            assert reducer.dimred_model_name == "new_dimred:3"

    def test_handle_duplicate_model_update(self):
        """Test handling duplicate model update notifications with version"""
        # Create the patch for MLflowClient and redis before importing anything
        with (
            patch("src.utils.mlflow_utils.MLflowClient") as mlflow_client_mock,
            patch(
                "src.arroyo_reduction.redis_model_store.RedisModelStore"
            ) as redis_mock,
            patch(
                "src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates"
            ),
            patch("src.arroyo_reduction.reducer.logger"),
        ):

            # Configure MLflowClient mock
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client

            # Set up mock store - return identifiers with versions
            mock_store = MagicMock()
            redis_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder:3"
            mock_store.get_dimred_model.return_value = "test_dimred:2"

            # Import the real class
            import sys
            if "src.arroyo_reduction.reducer" in sys.modules:
                del sys.modules["src.arroyo_reduction.reducer"]

            from src.arroyo_reduction.reducer import LatentSpaceReducer

            # Create the reducer
            reducer = LatentSpaceReducer()

            # Add mlflow_client attribute
            reducer.mlflow_client = mock_mlflow_client

            # Reset mock to clear call history
            mock_mlflow_client.reset_mock()

            # Create duplicate update with version (same as current)
            duplicate_update = {
                "model_type": "autoencoder",
                "model_name": "test_autoencoder:3",
            }

            # Track the initial state
            initial_autoencoder = reducer.autoencoder_model_name
            initial_dimred = reducer.dimred_model_name

            # Handle duplicate update
            reducer._handle_model_update(duplicate_update)

            # Key assertions for deduplication:
            # 1. Model names should not change
            assert reducer.autoencoder_model_name == initial_autoencoder
            assert reducer.dimred_model_name == initial_dimred

            # 2. Most importantly, load_model should NOT be called for duplicates
            mock_mlflow_client.load_model.assert_not_called()

    def test_loading_flags_during_model_update(self):
        """Test that loading flags are set and reset correctly during model update"""
        # Create the patch for MLflowClient and redis before importing anything
        with (
            patch("src.utils.mlflow_utils.MLflowClient") as mlflow_client_mock,
            patch(
                "src.arroyo_reduction.redis_model_store.RedisModelStore"
            ) as redis_mock,
            patch(
                "src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates"
            ),
            patch("src.arroyo_reduction.reducer.logger"),
        ):

            # Configure MLflowClient mock
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client

            # Set up mock store
            mock_store = MagicMock()
            redis_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder"
            mock_store.get_dimred_model.return_value = "test_dimred"

            # Import the real class
            from src.arroyo_reduction.reducer import LatentSpaceReducer

            # Create the reducer
            reducer = LatentSpaceReducer()

            # Add mlflow_client attribute
            reducer.mlflow_client = mock_mlflow_client

            # Create test update message (can include version)
            update = {"model_type": "autoencoder", "model_name": "new_autoencoder:7"}

            # Mock the flags to verify they're being set correctly
            reducer.is_loading_model = False
            reducer.loading_model_type = None

            # Handle the update - this should set the flags to True during loading
            reducer._handle_model_update(update)

            # Verify flags are reset to False after loading completes
            assert reducer.is_loading_model == False
            assert reducer.loading_model_type == None

            # Test with exception during model loading
            mock_mlflow_client.load_model.side_effect = Exception("Test error")

            # Reset flags
            reducer.is_loading_model = False
            reducer.loading_model_type = None

            # Handle update with error - should still reset flags
            reducer._handle_model_update(update)

            # Verify flags are reset even after exception
            assert reducer.is_loading_model == False
            assert reducer.loading_model_type == None

    def test_subscribe_to_model_updates(self):
        """Test subscribing to model updates"""
        # Create a mock thread class and instance
        mock_thread = MagicMock()

        # Test with just the thread mocked
        with (
            patch("threading.Thread", return_value=mock_thread) as mock_thread_class,
            patch("src.arroyo_reduction.redis_model_store.RedisModelStore"),
            patch("src.utils.mlflow_utils.MLflowClient"),
        ):

            # Import the real class but patch the __init__ to avoid complex initialization
            from src.arroyo_reduction.reducer import LatentSpaceReducer

            with patch.object(LatentSpaceReducer, "__init__", return_value=None):
                # Create the reducer without calling __init__
                reducer = LatentSpaceReducer()

                # Mock any necessary attributes
                reducer.autoencoder_model_name = "test_autoencoder"
                reducer.dimred_model_name = "test_dimred"

                # Call the method directly
                reducer._subscribe_to_model_updates()

                # Verify thread was created with expected parameters
                mock_thread_class.assert_called_once()
                args, kwargs = mock_thread_class.call_args
                assert kwargs["daemon"] is True
                assert "target" in kwargs

                # Verify thread was started
                mock_thread.start.assert_called_once()