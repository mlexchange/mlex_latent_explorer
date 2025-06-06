import pytest
from unittest.mock import patch, MagicMock
import json
from src.callbacks.execute import (
    store_dialog_models_in_redis, 
    store_sidebar_models_in_redis
)

class TestModelCallbacks:
    
    @pytest.fixture
    def mock_redis(self):
        """Mock the Redis client"""
        with patch('src.callbacks.execute.redis_client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_mlflow_client(self):
        """Mock the MLflowClient"""
        with patch('src.callbacks.execute.mlflow_client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_time(self):
        """Mock the time module"""
        with patch('src.callbacks.execute.import_time_module') as mock_import_time:
            mock_time = MagicMock()
            mock_time.time.return_value = 1234567890
            mock_import_time.return_value = mock_time
            yield mock_time
    
    def test_store_dialog_models_in_redis(self, mock_redis, mock_time):
        """Test storing models from dialog in Redis"""
        # Test data
        autoencoder = "dialog_autoencoder"
        dimred = "dialog_dimred"
        counter = 5
        
        # Call the function
        result = store_dialog_models_in_redis(autoencoder, dimred, counter)
        
        # Verify counter is incremented
        assert result == 6
        
        # Verify Redis set was called for both models
        mock_redis.set.assert_any_call("selected_mlflow_model", autoencoder)
        mock_redis.set.assert_any_call("selected_dim_reduction_model", dimred)
        
        # Verify publish was called for both models with correct messages
        assert mock_redis.publish.call_count == 2
        
        # Check first publish call (autoencoder)
        first_call = mock_redis.publish.call_args_list[0]
        assert first_call[0][0] == "model_updates"
        message1 = json.loads(first_call[0][1])
        assert message1["model_type"] == "autoencoder"
        assert message1["model_name"] == autoencoder
        
        # Check second publish call (dimred)
        second_call = mock_redis.publish.call_args_list[1]
        assert second_call[0][0] == "model_updates"
        message2 = json.loads(second_call[0][1])
        assert message2["model_type"] == "dimred"
        assert message2["model_name"] == dimred
    
    def test_store_sidebar_models_in_redis(self, mock_redis, mock_time):
        """Test storing models from sidebar in Redis"""
        # Test data
        autoencoder = "sidebar_autoencoder"
        dimred = "sidebar_dimred"
        counter = 10
        
        # Call the function
        result = store_sidebar_models_in_redis(autoencoder, dimred, counter)
        
        # Verify counter is incremented
        assert result == 11
        
        # Verify Redis set was called for both models
        mock_redis.set.assert_any_call("selected_mlflow_model", autoencoder)
        mock_redis.set.assert_any_call("selected_dim_reduction_model", dimred)
        
        # Verify publish was called for both models
        assert mock_redis.publish.call_count == 2
    
    def test_handle_redis_unavailable(self, mock_redis, mock_time):
        """Test behavior when Redis is unavailable"""
        # Set redis_client to None
        with patch('src.callbacks.execute.redis_client', None):
            # Call function
            counter = 5
            result = store_dialog_models_in_redis("model1", "model2", counter)
            
            # Verify counter is returned unchanged
            assert result == counter