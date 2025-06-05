import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Create patches at the module level
redis_mock_patch = patch('src.arroyo_reduction.redis_model_store.RedisModelStore')
mlflow_client_mock_patch = patch('src.utils.mlflow_utils.MLflowClient')
pil_mock_patch = patch('PIL.Image.fromarray')

class TestReducer:
    
    @pytest.fixture
    def mock_event(self):
        """Create a mock event with image data"""
        event = MagicMock()
        # Create test image data (128x128 random array)
        event.image.array = np.random.rand(128, 128).astype(np.float32)
        return event
    
    @pytest.fixture
    def setup_mocks(self):
        """Set up and start all the mocks"""
        # Start all the patches
        redis_mock = redis_mock_patch.start()
        mlflow_client_mock = mlflow_client_mock_patch.start()
        pil_mock = pil_mock_patch.start()
        
        # Configure the Redis mock
        mock_store = MagicMock()
        redis_mock.return_value = mock_store
        mock_store.get_autoencoder_model.return_value = "test_autoencoder"
        mock_store.get_dimred_model.return_value = "test_dimred"
        
        # Configure the MLFlow mock
        mock_mlflow_client = MagicMock()
        mlflow_client_mock.return_value = mock_mlflow_client
        
        # Configure models to return test data
        latent_features = np.random.rand(1, 512).astype(np.float32)
        umap_coords = np.random.rand(1, 2).astype(np.float32)
        
        mock_autoencoder = MagicMock()
        mock_dimred = MagicMock()
        
        mock_autoencoder.predict.return_value = {"latent_features": latent_features}
        mock_dimred.predict.return_value = {"umap_coords": umap_coords}
        
        # Configure the load_model method to return appropriate models
        mock_mlflow_client.load_model.side_effect = lambda model_name: mock_autoencoder if model_name == "test_autoencoder" else mock_dimred
        
        # Configure PIL mock
        mock_pil_image = MagicMock()
        pil_mock.return_value = mock_pil_image
        
        # Store all mocks and test data
        mocks = {
            "redis": redis_mock,
            "store": mock_store,
            "mlflow_client": mock_mlflow_client,
            "autoencoder": mock_autoencoder,
            "dimred": mock_dimred,
            "pil": pil_mock,
            "pil_image": mock_pil_image,
            "latent_features": latent_features,
            "umap_coords": umap_coords
        }
        
        yield mocks
        
        # Stop all patches after the test
        redis_mock_patch.stop()
        mlflow_client_mock_patch.stop()
        pil_mock_patch.stop()
    
    @pytest.fixture
    def reducer(self, setup_mocks):
        """Create a LatentSpaceReducer with all dependencies mocked"""
        # Patch _subscribe_to_model_updates to avoid threading issues
        with patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'):
            # Import here after mocks are set up
            from src.arroyo_reduction.reducer import LatentSpaceReducer
            
            # Create the reducer with all dependencies mocked
            reducer = LatentSpaceReducer()
            
            # Set references to the mock models
            reducer.current_torch_model = setup_mocks["autoencoder"]
            reducer.current_dim_reduction_model = setup_mocks["dimred"]
            
            # Mock the transform method to return a PyTorch tensor
            test_tensor = torch.rand(1, 128, 128)
            mock_transform = MagicMock()
            mock_transform.return_value = test_tensor
            reducer.current_transform = mock_transform
            
            # Store test data
            reducer._test_data = {
                "tensor": test_tensor,
                "mocks": setup_mocks
            }
            
            return reducer
    
    def test_reduce(self, reducer, mock_event):
        """Test that reduce() correctly processes an image through models"""
        # Get the test data and mocks
        test_data = reducer._test_data
        mocks = test_data["mocks"]
        
        # Create real numpy arrays with proper dimensions and types
        latent_features = np.random.rand(1, 512).astype(np.float32)
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
        
        # 1. PIL.Image.fromarray should be called
        mocks["pil"].assert_called()
        
        # 2. The transform should be called
        reducer.current_transform.assert_called()
        
        # 3. The autoencoder should be called
        mocks["autoencoder"].predict.assert_called()
        
        # 4. The dimred model should be called
        mocks["dimred"].predict.assert_called()
        
        # 5. The result should be the actual numpy array from the mocked umap_coords
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)  # Check expected shape of UMAP coordinates
        # Verify it's actually the same array we created
        np.testing.assert_array_equal(result, umap_coords)
    
    def test_init_loads_models_from_redis(self):
        """Test that constructor loads models from Redis"""
        # Create mocks for dependencies
        with patch('src.arroyo_reduction.redis_model_store.RedisModelStore', autospec=True) as redis_class_mock, \
             patch('src.utils.mlflow_utils.MLflowClient') as mlflow_client_mock, \
             patch('src.arroyo_reduction.reducer.LatentSpaceReducer._subscribe_to_model_updates'):
            
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