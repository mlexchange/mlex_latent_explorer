import pytest
import redis
import json
from unittest.mock import patch, MagicMock
from src.arroyo_reduction.redis_model_store import RedisModelStore

class TestRedisStore:
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client"""
        return MagicMock()
    
    @pytest.fixture
    def redis_store(self, mock_redis_client):
        """Create a RedisModelStore with a mock Redis client"""
        # Patch redis.Redis to return our mock
        with patch('redis.Redis', return_value=mock_redis_client):
            # Create the store - this will use our patched Redis
            store = RedisModelStore(host="localhost", port=6379)
            
            # Make sure our mock was used
            assert store.redis_client is mock_redis_client
            
            # Store the mock for tests to use
            store._mock_client = mock_redis_client
            
            yield store
    
    def test_store_and_get_models(self, redis_store):
        """Test storing and retrieving model names"""
        # Test data
        autoencoder_name = "test_autoencoder_model"
        dimred_name = "test_dimred_model"
        
        # Configure mock to return our test data
        redis_store._mock_client.get.side_effect = [autoencoder_name, dimred_name]
        
        # Store models
        redis_store.store_autoencoder_model(autoencoder_name)
        redis_store.store_dimred_model(dimred_name)
        
        # Verify Redis set was called
        redis_store._mock_client.set.assert_any_call(
            RedisModelStore.KEY_AUTOENCODER_MODEL, 
            autoencoder_name
        )
        redis_store._mock_client.set.assert_any_call(
            RedisModelStore.KEY_DIMRED_MODEL, 
            dimred_name
        )
        
        # Get models and verify
        retrieved_autoencoder = redis_store.get_autoencoder_model()
        retrieved_dimred = redis_store.get_dimred_model()
        
        assert retrieved_autoencoder == autoencoder_name
        assert retrieved_dimred == dimred_name
        
        # Verify get was called
        redis_store._mock_client.get.assert_any_call(RedisModelStore.KEY_AUTOENCODER_MODEL)
        redis_store._mock_client.get.assert_any_call(RedisModelStore.KEY_DIMRED_MODEL)
    
    def test_publish_model_update(self, redis_store):
        """Test pub/sub functionality"""
        model_type = "autoencoder"
        model_name = "new_model"
        
        # Mock publish to return 1 (one client received the message)
        redis_store._mock_client.publish.return_value = 1
        
        # Publish an update
        result = redis_store.publish_model_update(model_type, model_name)
        
        # Verify result
        assert result is True
        
        # Verify publish was called
        redis_store._mock_client.publish.assert_called_once()
        
        # Check channel and message format
        args = redis_store._mock_client.publish.call_args[0]
        assert args[0] == RedisModelStore.CHANNEL_MODEL_UPDATES
        
        # Parse and check message
        message = json.loads(args[1])
        assert message["model_type"] == model_type
        assert message["model_name"] == model_name
        assert "timestamp" in message
        
    def test_subscribe_to_model_updates(self, redis_store):
        """Test subscribing to model updates"""
        # Set up mocks
        mock_thread = MagicMock()
        mock_thread_cls = MagicMock(return_value=mock_thread)
        
        # Set up callback
        callback = MagicMock()
        
        # Call the function
        with patch('threading.Thread', mock_thread_cls):
            redis_store.subscribe_to_model_updates(callback)
            
            # Verify thread was created and started
            mock_thread_cls.assert_called_once()
            assert mock_thread_cls.call_args[1]["daemon"] is True
            
            # Get the thread target function
            thread_target = mock_thread_cls.call_args[1]["target"]
            assert callable(thread_target)
            
            # Verify thread was started
            mock_thread.start.assert_called_once()