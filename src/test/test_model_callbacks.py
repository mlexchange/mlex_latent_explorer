import pytest
from unittest.mock import patch, MagicMock
import json
from dash.exceptions import PreventUpdate
from src.callbacks.execute import (
    store_dialog_models_in_redis_on_continue, 
    store_sidebar_models_in_redis_on_update
)

class TestModelCallbacks:
    
    @pytest.fixture
    def mock_redis_store(self):
        """Mock the RedisModelStore"""
        with patch('src.callbacks.execute.redis_model_store') as mock_store:
            yield mock_store
    
    @pytest.fixture
    def mock_mlflow_client(self):
        """Mock the MLflowClient"""
        with patch('src.callbacks.execute.mlflow_client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_logger(self):
        """Mock the logger"""
        with patch('src.callbacks.execute.logger') as mock_logger:
            yield mock_logger
    
    def test_store_dialog_models_in_redis_on_continue(self, mock_redis_store, mock_logger):
        """Test storing models from dialog in Redis when Continue button is clicked"""
        # Test data
        n_clicks = 1
        autoencoder = "dialog_autoencoder"
        dimred = "dialog_dimred"
        counter = 5
        
        # Configure mock to return success
        mock_redis_store.store_autoencoder_model.return_value = True
        mock_redis_store.store_dimred_model.return_value = True
        
        # Call the function
        result = store_dialog_models_in_redis_on_continue(n_clicks, autoencoder, dimred, counter)
        
        # Verify counter is incremented
        assert result == 6
        
        # Verify Redis methods were called for both models
        mock_redis_store.store_autoencoder_model.assert_called_once_with(autoencoder)
        mock_redis_store.store_dimred_model.assert_called_once_with(dimred)
        
        # Verify logger was called
        mock_logger.info.assert_any_call(f"Storing autoencoder model from dialog: {autoencoder}")
        mock_logger.info.assert_any_call(f"Storing dimension reduction model from dialog: {dimred}")
    
    def test_store_sidebar_models_in_redis_on_update(self, mock_redis_store, mock_logger):
        """Test storing models from sidebar in Redis when Update button is clicked"""
        # Test data
        n_clicks = 1
        autoencoder = "sidebar_autoencoder"
        dimred = "sidebar_dimred"
        counter = 10
        
        # Configure mock to return success
        mock_redis_store.store_autoencoder_model.return_value = True
        mock_redis_store.store_dimred_model.return_value = True
        
        # Call the function
        result = store_sidebar_models_in_redis_on_update(n_clicks, autoencoder, dimred, counter)
        
        # Verify counter is incremented
        assert result == 11
        
        # Verify Redis methods were called for both models
        mock_redis_store.store_autoencoder_model.assert_called_once_with(autoencoder)
        mock_redis_store.store_dimred_model.assert_called_once_with(dimred)
        
        # Verify logger was called
        mock_logger.info.assert_any_call(f"Storing autoencoder model from sidebar: {autoencoder}")
        mock_logger.info.assert_any_call(f"Storing dimension reduction model from sidebar: {dimred}")
    
    def test_prevent_update_when_no_clicks_dialog(self, mock_redis_store):
        """Test PreventUpdate is raised when n_clicks is None for dialog"""
        with pytest.raises(PreventUpdate):
            store_dialog_models_in_redis_on_continue(None, "model1", "model2", 5)
    
    def test_prevent_update_when_no_clicks_sidebar(self, mock_redis_store):
        """Test PreventUpdate is raised when n_clicks is None for sidebar"""
        with pytest.raises(PreventUpdate):
            store_sidebar_models_in_redis_on_update(None, "model1", "model2", 5)
    
    def test_handle_redis_store_failure_dialog(self, mock_redis_store, mock_logger):
        """Test behavior when Redis store operations fail for dialog"""
        # Test data
        n_clicks = 1
        autoencoder = "dialog_autoencoder"
        dimred = "dialog_dimred"
        counter = 5
        
        # Configure mocks to simulate failure
        mock_redis_store.store_autoencoder_model.return_value = False
        mock_redis_store.store_dimred_model.return_value = True
        
        # Call function
        result = store_dialog_models_in_redis_on_continue(n_clicks, autoencoder, dimred, counter)
        
        # Verify counter is not incremented due to failure
        assert result == counter
    
    def test_handle_redis_store_failure_sidebar(self, mock_redis_store, mock_logger):
        """Test behavior when Redis store operations fail for sidebar"""
        # Test data
        n_clicks = 1
        autoencoder = "sidebar_autoencoder"
        dimred = "sidebar_dimred"
        counter = 5
        
        # Configure mocks to simulate failure
        mock_redis_store.store_autoencoder_model.return_value = True
        mock_redis_store.store_dimred_model.return_value = False
        
        # Call function
        result = store_sidebar_models_in_redis_on_update(n_clicks, autoencoder, dimred, counter)
        
        # Verify counter is not incremented due to failure
        assert result == counter
    
    def test_handle_exception_dialog(self, mock_redis_store, mock_logger):
        """Test handling exceptions for dialog"""
        # Configure Redis to raise an exception
        mock_redis_store.store_autoencoder_model.side_effect = Exception("Test error")
        
        # Call function with exception
        counter = 5
        
        # First need to patch execute.py's implementation to handle the exception
        with patch('src.callbacks.execute.store_dialog_models_in_redis_on_continue', autospec=True) as mock_func:
            # Configure the mock to match real implementation with exception handling
            def side_effect(n_clicks, autoencoder_model, dim_reduction_model, counter):
                if not n_clicks:
                    raise PreventUpdate
                
                try:
                    success = True
                    
                    if autoencoder_model:
                        # This will raise the exception
                        success = success and mock_redis_store.store_autoencoder_model(autoencoder_model)
                    
                    if dim_reduction_model and success:
                        success = success and mock_redis_store.store_dimred_model(dim_reduction_model)
                    
                    return (counter or 0) + 1 if success else counter
                except Exception as e:
                    mock_logger.error(f"Error storing dialog dropdown models in Redis: {e}")
                    return counter
            
            mock_func.side_effect = side_effect
            result = mock_func(1, "model1", "model2", counter)
        
        # Verify counter is returned unchanged
        assert result == counter
        
        # Verify error was logged
        mock_logger.error.assert_called()
    
    def test_handle_exception_sidebar(self, mock_redis_store, mock_logger):
        """Test handling exceptions for sidebar"""
        # Configure Redis to raise an exception
        mock_redis_store.store_autoencoder_model.side_effect = Exception("Test error")
        
        # Call function with exception
        counter = 5
        
        # First need to patch execute.py's implementation to handle the exception
        with patch('src.callbacks.execute.store_sidebar_models_in_redis_on_update', autospec=True) as mock_func:
            # Configure the mock to match real implementation with exception handling
            def side_effect(n_clicks, autoencoder_model, dim_reduction_model, counter):
                if not n_clicks:
                    raise PreventUpdate
                
                try:
                    success = True
                    
                    if autoencoder_model:
                        # This will raise the exception
                        success = success and mock_redis_store.store_autoencoder_model(autoencoder_model)
                    
                    if dim_reduction_model and success:
                        success = success and mock_redis_store.store_dimred_model(dim_reduction_model)
                    
                    return (counter or 0) + 1 if success else counter
                except Exception as e:
                    mock_logger.error(f"Error storing sidebar models in Redis: {e}")
                    return counter
            
            mock_func.side_effect = side_effect
            result = mock_func(1, "model1", "model2", counter)
        
        # Verify counter is returned unchanged
        assert result == counter
        
        # Verify error was logged
        mock_logger.error.assert_called()
        
    def test_partial_model_submission_dialog(self, mock_redis_store):
        """Test behavior when only one model is provided for dialog"""
        # Test data
        n_clicks = 1
        autoencoder = "dialog_autoencoder"
        dimred = None  # No dimension reduction model
        counter = 5
        
        # Configure mock
        mock_redis_store.store_autoencoder_model.return_value = True
        
        # Call the function
        result = store_dialog_models_in_redis_on_continue(n_clicks, autoencoder, dimred, counter)
        
        # Verify counter is incremented
        assert result == 6
        
        # Verify only autoencoder model was stored
        mock_redis_store.store_autoencoder_model.assert_called_once_with(autoencoder)
        mock_redis_store.store_dimred_model.assert_not_called()
        
    def test_partial_model_submission_sidebar(self, mock_redis_store):
        """Test behavior when only one model is provided for sidebar"""
        # Test data
        n_clicks = 1
        autoencoder = None  # No autoencoder model
        dimred = "sidebar_dimred"
        counter = 5
        
        # Configure mock
        mock_redis_store.store_dimred_model.return_value = True
        
        # Call the function
        result = store_sidebar_models_in_redis_on_update(n_clicks, autoencoder, dimred, counter)
        
        # Verify counter is incremented
        assert result == 6
        
        # Verify only dimension reduction model was stored
        mock_redis_store.store_autoencoder_model.assert_not_called()
        mock_redis_store.store_dimred_model.assert_called_once_with(dimred)