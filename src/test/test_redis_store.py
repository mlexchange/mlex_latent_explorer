import json
from unittest.mock import MagicMock, patch

import pytest
import redis

from src.arroyo_reduction.redis_model_store import RedisModelStore
from src.test.test_utils import mock_redis_client, redis_test_store


class TestRedisStore:

    def test_store_and_get_models(self, redis_test_store):
        """Test storing and retrieving model names with versions"""
        # Test data - now with version format
        autoencoder_name = "test_autoencoder_model:3"
        dimred_name = "test_dimred_model:2"

        # Configure mock to return our test data
        redis_test_store._mock_client.get.side_effect = [autoencoder_name, dimred_name]

        # Store models
        redis_test_store.store_autoencoder_model(autoencoder_name)
        redis_test_store.store_dimred_model(dimred_name)

        # Verify Redis set was called
        redis_test_store._mock_client.set.assert_any_call(
            RedisModelStore.KEY_AUTOENCODER_MODEL, autoencoder_name
        )
        redis_test_store._mock_client.set.assert_any_call(
            RedisModelStore.KEY_DIMRED_MODEL, dimred_name
        )

        # Get models and verify
        retrieved_autoencoder = redis_test_store.get_autoencoder_model()
        retrieved_dimred = redis_test_store.get_dimred_model()

        assert retrieved_autoencoder == autoencoder_name
        assert retrieved_dimred == dimred_name

        # Verify get was called
        redis_test_store._mock_client.get.assert_any_call(
            RedisModelStore.KEY_AUTOENCODER_MODEL
        )
        redis_test_store._mock_client.get.assert_any_call(
            RedisModelStore.KEY_DIMRED_MODEL
        )

    def test_publish_model_update(self, redis_test_store):
        """Test pub/sub functionality with versions"""
        model_type = "autoencoder"
        model_name = "new_model:5"

        # Mock publish to return 1 (one client received the message)
        redis_test_store._mock_client.publish.return_value = 1

        # Publish an update
        result = redis_test_store.publish_model_update(model_type, model_name)

        # Verify result
        assert result is True

        # Verify publish was called
        redis_test_store._mock_client.publish.assert_called_once()

        # Check channel and message format
        args = redis_test_store._mock_client.publish.call_args[0]
        assert args[0] == RedisModelStore.CHANNEL_MODEL_UPDATES

        # Parse and check message
        message = json.loads(args[1])
        assert message["model_type"] == model_type
        assert message["model_name"] == model_name
        assert "timestamp" in message

    def test_subscribe_to_model_updates(self, redis_test_store):
        """Test subscribing to model updates"""
        # Set up mocks
        mock_thread = MagicMock()
        mock_thread_cls = MagicMock(return_value=mock_thread)

        # Set up callback
        callback = MagicMock()

        # Call the function
        with patch("threading.Thread", mock_thread_cls):
            redis_test_store.subscribe_to_model_updates(callback)

            # Verify thread was created and started
            mock_thread_cls.assert_called_once()
            assert mock_thread_cls.call_args[1]["daemon"] is True

            # Get the thread target function
            thread_target = mock_thread_cls.call_args[1]["target"]
            assert callable(thread_target)

            # Verify thread was started
            mock_thread.start.assert_called_once()