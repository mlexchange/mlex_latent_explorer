# src/test/test_model_callbacks.py
from unittest.mock import MagicMock, patch

import pytest
from dash.exceptions import PreventUpdate

from src.callbacks.live_mode import (
    handle_model_continue,
    reset_update_button,
    update_live_models,
)
from src.test.test_utils import (
    mock_live_mode_mlflow_client,
    mock_logger,
    mock_redis_store,
)


class TestModelCallbacks:

    def test_handle_model_continue(
        self, mock_redis_store, mock_live_mode_mlflow_client
    ):
        """Test handling continue button with compatible models"""
        # Setup
        continue_clicks = 1
        experiment_name = "test_experiment"  # ADD THIS LINE
        selected_auto = "compatible_auto"
        selected_dimred = "compatible_dimred"
        auto_version = "3"
        dimred_version = "2"
        auto_options = [{"label": "Auto1", "value": "compatible_auto"}]
        dimred_options = [{"label": "Dimred1", "value": "compatible_dimred"}]

        # Configure compatibility check
        mock_live_mode_mlflow_client.check_model_compatibility.return_value = True
        
        # Mock get_model_versions to return version options
        mock_live_mode_mlflow_client.get_model_versions.return_value = [
            {"label": "Version 3", "value": "3"}
        ]

        # Mock plot functions to avoid errors
        with (
            patch("src.callbacks.live_mode.plot_empty_scatter", return_value={}),
            patch("src.callbacks.live_mode.plot_empty_heatmap", return_value={}),
        ):

            # Execute
            results = handle_model_continue(
                continue_clicks,
                experiment_name,  # ADD THIS LINE
                selected_auto,
                auto_version,
                selected_dimred,
                dimred_version,
                auto_options,
                dimred_options,
            )

        # Main verification: Redis calls with version identifiers
        expected_auto = f"{selected_auto}:{auto_version}"
        expected_dimred = f"{selected_dimred}:{dimred_version}"
        
        mock_redis_store.store_autoencoder_model.assert_called_once_with(expected_auto)
        mock_redis_store.store_dimred_model.assert_called_once_with(expected_dimred)
        mock_redis_store.store_experiment_name.assert_called_once_with(experiment_name)  # ADD THIS LINE
        
        # Verify dialog closed (first result should be False)
        assert results[0] is False
        
        # Verify selected_models dict is at index 1 with versions
        selected_models = results[1]
        assert selected_models["autoencoder"] == selected_auto
        assert selected_models["autoencoder_version"] == auto_version
        assert selected_models["dimred"] == selected_dimred
        assert selected_models["dimred_version"] == dimred_version
        assert selected_models["experiment_name"] == experiment_name  # ADD THIS LINE

    def test_handle_model_continue_incompatible(
        self, mock_redis_store, mock_live_mode_mlflow_client
    ):
        """Test handling continue button with incompatible models"""
        # Setup
        continue_clicks = 1
        experiment_name = "test_experiment"  # ADD THIS LINE
        selected_auto = "incompatible"
        selected_dimred = "compatible_dimred"
        auto_version = "1"
        dimred_version = "1"
        auto_options = [{"label": "Auto1", "value": "incompatible"}]
        dimred_options = [{"label": "Dimred1", "value": "compatible_dimred"}]

        # Configure compatibility check
        mock_live_mode_mlflow_client.check_model_compatibility.return_value = False

        # Execute
        result = handle_model_continue(
            continue_clicks,
            experiment_name,  # ADD THIS LINE
            selected_auto,
            auto_version,
            selected_dimred,
            dimred_version,
            auto_options,
            dimred_options,
        )

        # Verify Redis calls were not made
        mock_redis_store.store_autoencoder_model.assert_not_called()
        mock_redis_store.store_dimred_model.assert_not_called()
        mock_redis_store.store_experiment_name.assert_not_called()  # ADD THIS LINE

    def test_update_live_models(self, mock_redis_store, mock_live_mode_mlflow_client):
        """Test updating models from sidebar"""
        # Setup
        n_clicks = 1
        autoencoder = "test_autoencoder"
        dimred = "test_dimred"
        auto_version = "5"
        dimred_version = "3"
        data_project_dict = {"root_uri": "", "datasets": []}

        # Configure compatibility check
        mock_live_mode_mlflow_client.check_model_compatibility.return_value = True

        # Mock plot functions to avoid errors
        with (
            patch("src.callbacks.live_mode.plot_empty_scatter", return_value={}),
            patch("src.callbacks.live_mode.plot_empty_heatmap", return_value={}),
        ):

            # Execute
            results = update_live_models(
                n_clicks, autoencoder, auto_version, dimred, dimred_version, data_project_dict
            )

        # Main verification: Redis calls with version identifiers
        expected_auto = f"{autoencoder}:{auto_version}"
        expected_dimred = f"{dimred}:{dimred_version}"
        
        mock_redis_store.store_autoencoder_model.assert_called_once_with(expected_auto)
        mock_redis_store.store_dimred_model.assert_called_once_with(expected_dimred)
        
        # Verify selected_models is at index 0 with versions
        selected_models = results[0]
        assert selected_models["autoencoder"] == autoencoder
        assert selected_models["autoencoder_version"] == auto_version
        assert selected_models["dimred"] == dimred
        assert selected_models["dimred_version"] == dimred_version
        
        # Verify data_project_dict at index 1 contains live_models
        data_proj = results[1]
        assert "live_models" in data_proj
        assert data_proj["live_models"] == selected_models

    def test_update_live_models_failure(
        self, mock_redis_store, mock_live_mode_mlflow_client, mock_logger
    ):
        """Test error handling when Redis operations fail"""
        # Setup
        n_clicks = 1
        autoencoder = "test_autoencoder"
        dimred = "test_dimred"
        auto_version = "2"
        dimred_version = "1"
        data_project_dict = {"root_uri": "", "datasets": []}

        # Configure Redis failure
        mock_redis_store.store_autoencoder_model.side_effect = Exception("Redis error")

        # Execute
        with (
            patch("src.callbacks.live_mode.plot_empty_scatter", return_value={}),
            patch("src.callbacks.live_mode.plot_empty_heatmap", return_value={}),
        ):

            results = update_live_models(
                n_clicks, autoencoder, auto_version, dimred, dimred_version, data_project_dict
            )

        # Verify error handling - index 2 is button color, index 3 is button text
        assert results[2] == "danger"  # Button color
        assert results[3] == "Update Fail"  # Button text

        # Verify logger was called with error
        mock_logger.error.assert_called()

    def test_reset_update_button(self, mock_live_mode_mlflow_client):
        """Test resetting the update button state"""
        # Setup
        autoencoder = "test_autoencoder"
        dimred = "test_dimred"
        auto_version = "3"
        dimred_version = "2"
        selected_models = {
            "autoencoder": "old_autoencoder",
            "dimred": "old_dimred",
            "autoencoder_version": "1",
            "dimred_version": "1"
        }

        # Configure compatibility check
        mock_live_mode_mlflow_client.check_model_compatibility.return_value = True

        # Execute - models changed (different from selected_models)
        result_disabled, result_color, result_text = reset_update_button(
            autoencoder, auto_version, dimred, dimred_version, selected_models
        )

        # Verify - button should be enabled when models differ
        assert result_disabled is False
        assert result_color == "primary"
        assert result_text == "Update Models"

        # Execute - models unchanged (same name and version)
        selected_models_same = {
            "autoencoder": "test_autoencoder",
            "dimred": "test_dimred",
            "autoencoder_version": "3",
            "dimred_version": "2"
        }
        result_disabled, result_color, result_text = reset_update_button(
            autoencoder, auto_version, dimred, dimred_version, selected_models_same
        )

        # Verify - button should be disabled when nothing changed
        assert result_disabled is True
        assert result_color == "secondary"
        assert result_text == "Updated"

        # Execute - incompatible models
        mock_live_mode_mlflow_client.check_model_compatibility.return_value = False
        result_disabled, result_color, result_text = reset_update_button(
            "incompatible", auto_version, dimred, dimred_version, selected_models
        )

        # Verify - based on actual implementation behavior
        assert result_color == "danger"
        assert result_text == "Incompatible Models"