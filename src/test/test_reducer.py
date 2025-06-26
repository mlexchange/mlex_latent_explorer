import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.test.test_utils import mock_event, redis_mlflow_mocks

class TestReducer:
    
    @pytest.fixture
    def reducer(self, redis_mlflow_mocks):
        """Create a LatentSpaceReducer with all dependencies mocked"""
        # Patch _subscribe_to_model_updates to avoid threading issues
        with patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'):
            # Import here after mocks are set up
            from src.arroyo_reduction.reducer import LatentSpaceReducer
            
            # Create the reducer with all dependencies mocked
            reducer = LatentSpaceReducer()
            
            # Set references to the mock models
            reducer.current_torch_model = redis_mlflow_mocks["autoencoder"]
            reducer.current_dim_reduction_model = redis_mlflow_mocks["dimred"]
            
            # Store test data
            reducer._test_data = {
                "mocks": redis_mlflow_mocks
            }
            
            return reducer
    
    def test_reduce(self, reducer, mock_event, redis_mlflow_mocks):
        """Test that reduce() correctly processes an image through models"""
        # Get the mocks
        mocks = redis_mlflow_mocks
        
        # Create real numpy arrays with proper dimensions and types
        latent_features = np.random.rand(1, 64).astype(np.float32)
        umap_coords = np.random.rand(1, 2).astype(np.float32)
        
        # Set up the mocks to return real numpy arrays
        mocks["autoencoder"].predict.return_value = {"latent_features": latent_features}
        mocks["dimred"].predict.return_value = {"umap_coords": umap_coords}
        
        # Set the models
        reducer.current_torch_model = mocks["autoencoder"]
        reducer.current_dim_reduction_model = mocks["dimred"]
        
        # Mock logger to prevent logging issues
        with patch('src.arroyo_reduction.reducer.logger'):
            # Call reduce()
            result = reducer.reduce(mock_event)
        
        # Verify the processing flow
        
        # 1. The autoencoder should be called with the correct input
        mocks["autoencoder"].predict.assert_called_once()
        
        # 2. The dimred model should be called with the latent features
        mocks["dimred"].predict.assert_called_once_with(latent_features)
        
        # 3. The result should be the actual numpy array from the mocked umap_coords
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)  # Check expected shape of UMAP coordinates
        # Verify it's actually the same array we created
        np.testing.assert_array_equal(result, umap_coords)
    
    def test_reduce_during_model_loading(self, reducer, mock_event):
        """Test that reduce() returns zeros when models are loading"""
        # Set loading flag to True to simulate model loading
        reducer.is_loading_model = True
        reducer.loading_model_type = "autoencoder"
        
        # Mock logger to prevent logging issues
        with patch('src.arroyo_reduction.reducer.logger'):
            # Call reduce()
            result = reducer.reduce(mock_event)
        
        # Verify result is zeros of correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)
        assert np.all(result == 0)
    
    def test_init_loads_models_from_redis(self):
        """Test that constructor loads models from Redis"""
        # Create mocks for dependencies
        with patch('src.arroyo_reduction.redis_model_store.RedisModelStore', autospec=True) as redis_class_mock, \
             patch('src.utils.mlflow_utils.MLflowClient') as mlflow_client_mock, \
             patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'), \
             patch('src.arroyo_reduction.reducer.logger'):
            
            # Set up mock store - important to do this before importing
            mock_store = MagicMock()
            redis_class_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder"
            mock_store.get_dimred_model.return_value = "test_dimred"
            
            # Set up mock MLflowClient
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client
            
            # Set up mock models
            mock_autoencoder = MagicMock()
            mock_dimred = MagicMock()
            
            # Configure the load_model method
            mock_mlflow_client.load_model.side_effect = lambda name: mock_autoencoder if name == "test_autoencoder" else mock_dimred
            
            # Import the real class - this needs to be done AFTER setting up the mocks
            # to ensure the imports in the module use our mocked objects
            import sys
            if 'src.arroyo_reduction.reducer' in sys.modules:
                del sys.modules['src.arroyo_reduction.reducer']
            
            # Now import the module fresh
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
            
            # Verify models were loaded
            mock_mlflow_client.load_model.assert_any_call("test_autoencoder")
            mock_mlflow_client.load_model.assert_any_call("test_dimred")
            
            # Verify loading flags are set correctly during initialization
            assert not reducer.is_loading_model
            assert reducer.loading_model_type is None
    
    def test_handle_model_update(self):
        """Test handling model update notifications"""
        # Set up mock models
        mock_orig_autoencoder = MagicMock()
        mock_new_autoencoder = MagicMock()
        mock_orig_dimred = MagicMock()
        mock_new_dimred = MagicMock()
        
        # Create the patch for MLflowClient and redis before importing anything
        with patch('src.utils.mlflow_utils.MLflowClient') as mlflow_client_mock, \
             patch('src.arroyo_reduction.redis_model_store.RedisModelStore') as redis_mock, \
             patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'), \
             patch('src.arroyo_reduction.reducer.logger'):  # Add logger patch to suppress errors
                
            # Configure MLflowClient mock
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client
            
            # Configure load_model to return different models based on name
            mock_mlflow_client.load_model.side_effect = lambda name: {
                "test_autoencoder": mock_orig_autoencoder,
                "new_autoencoder": mock_new_autoencoder,
                "test_dimred": mock_orig_dimred,
                "new_dimred": mock_new_dimred
            }.get(name, MagicMock())
                    
            # Set up mock store
            mock_store = MagicMock()
            redis_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder"
            mock_store.get_dimred_model.return_value = "test_dimred"
            
            # Import the real class - ensure we have a fresh import
            import sys
            if 'src.arroyo_reduction.reducer' in sys.modules:
                del sys.modules['src.arroyo_reduction.reducer']
            
            from src.arroyo_reduction.reducer import LatentSpaceReducer
            
            # Create the reducer - which loads the initial models
            reducer = LatentSpaceReducer()
            
            # Add mlflow_client attribute with load_model method
            reducer.mlflow_client = mock_mlflow_client
            
            # Verify initial models were loaded
            assert reducer.autoencoder_model_name == "test_autoencoder"
            assert reducer.dimred_model_name == "test_dimred"
            
            # Create update messages
            autoencoder_update = {
                "model_type": "autoencoder",
                "model_name": "new_autoencoder"
            }
            
            # Handle autoencoder update
            reducer._handle_model_update(autoencoder_update)
            
            # Verify model name was updated
            assert reducer.autoencoder_model_name == "new_autoencoder"
            
            # Verify the mlflow_client.load_model was called with the new name
            mock_mlflow_client.load_model.assert_any_call("new_autoencoder")
            
            # Create update for dimred model
            dimred_update = {
                "model_type": "dimred",
                "model_name": "new_dimred"
            }
            
            # Handle dimred update
            reducer._handle_model_update(dimred_update)
            
            # Verify dimred model name was updated
            assert reducer.dimred_model_name == "new_dimred"
            
            # Verify mlflow_client.load_model was called with the new dimred name
            mock_mlflow_client.load_model.assert_any_call("new_dimred")
            
            # Test with invalid update
            invalid_update = {
                "model_type": "unknown",
                "model_name": "test"
            }
            reducer._handle_model_update(invalid_update)
            
            # Verify no changes with invalid update
            assert reducer.autoencoder_model_name == "new_autoencoder"
            assert reducer.dimred_model_name == "new_dimred"
    
    def test_handle_duplicate_model_update(self):
        """Test handling duplicate model update notifications"""
        # Create the patch for MLflowClient and redis before importing anything
        with patch('src.utils.mlflow_utils.MLflowClient') as mlflow_client_mock, \
             patch('src.arroyo_reduction.redis_model_store.RedisModelStore') as redis_mock, \
             patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'):
                
            # Configure MLflowClient mock
            mock_mlflow_client = MagicMock()
            mlflow_client_mock.return_value = mock_mlflow_client
            
            # Set up mock store
            mock_store = MagicMock()
            redis_mock.return_value = mock_store
            mock_store.get_autoencoder_model.return_value = "test_autoencoder"
            mock_store.get_dimred_model.return_value = "test_dimred"
            
            # Import the real class - ensure we have a fresh import
            import sys
            if 'src.arroyo_reduction.reducer' in sys.modules:
                del sys.modules['src.arroyo_reduction.reducer']
            
            from src.arroyo_reduction.reducer import LatentSpaceReducer
            
            # Create the reducer - which loads the initial models
            reducer = LatentSpaceReducer()
            
            # Add mlflow_client attribute with load_model method
            reducer.mlflow_client = mock_mlflow_client
            
            # Reset mock_mlflow_client to clear call history
            mock_mlflow_client.reset_mock()
            
            # Create duplicate update message (same as current model)
            duplicate_update = {
                "model_type": "autoencoder",
                "model_name": "test_autoencoder"  # Same as current model
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
            
            # Verify the model was NOT reloaded
            mock_mlflow_client.load_model.assert_not_called()
            
            # Verify model names were not changed
            assert reducer.autoencoder_model_name == "test_autoencoder"
            assert reducer.dimred_model_name == "test_dimred"
    
    def test_loading_flags_during_model_update(self):
        """Test that loading flags are set and reset correctly during model update"""
        # Create the patch for MLflowClient and redis before importing anything
        with patch('src.utils.mlflow_utils.MLflowClient') as mlflow_client_mock, \
             patch('src.arroyo_reduction.redis_model_store.RedisModelStore') as redis_mock, \
             patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'), \
             patch('src.arroyo_reduction.reducer.logger'):
                
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
            
            # Create test update message
            update = {
                "model_type": "autoencoder",
                "model_name": "new_autoencoder"
            }
            
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
        with patch('threading.Thread', return_value=mock_thread) as mock_thread_class, \
             patch('src.arroyo_reduction.redis_model_store.RedisModelStore'), \
             patch('src.utils.mlflow_utils.MLflowClient'):
            
            # Import the real class but patch the __init__ to avoid complex initialization
            from src.arroyo_reduction.reducer import LatentSpaceReducer
            
            with patch.object(LatentSpaceReducer, '__init__', return_value=None):
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
                assert kwargs['daemon'] is True
                assert 'target' in kwargs
                
                # Verify thread was started
                mock_thread.start.assert_called_once()