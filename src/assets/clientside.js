// Set this in your HTML or another script to configure log level
// window.DASH_LOG_LEVEL = 3; // INFO level by default

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    liveWS: {
        updateLiveData: function(message, buffer_data, n_clicks, data_project_dict, live_indices, selected_models) {
            // Initialize log level if not set
            window.DASH_LOG_LEVEL = window.DASH_LOG_LEVEL || 3;

            const LOG_LEVELS = {
                NONE: 0,
                ERROR: 1,
                WARN: 2,
                INFO: 3,
                DEBUG: 4
            };

            const log = {
                error: (msg, ...args) => window.DASH_LOG_LEVEL >= LOG_LEVELS.ERROR && console.error(msg, ...args),
                warn: (msg, ...args) => window.DASH_LOG_LEVEL >= LOG_LEVELS.WARN && console.warn(msg, ...args),
                info: (msg, ...args) => window.DASH_LOG_LEVEL >= LOG_LEVELS.INFO && console.log(msg, ...args),
                debug: (msg, ...args) => window.DASH_LOG_LEVEL >= LOG_LEVELS.DEBUG && console.log('[DEBUG]', msg, ...args)
            };

            if (n_clicks !== null && n_clicks % 2 === 1) {
                try {
                    log.info("Clientside callback triggered with message:", message);
                    log.debug("Current buffer_data:", buffer_data, "Type:", typeof buffer_data);
                    log.debug("Selected models:", selected_models);

                    // Initialize default values
                    if (!Array.isArray(buffer_data)) buffer_data = [];
                    if (Object.keys(data_project_dict).length === 0) {
                        data_project_dict = {
                            "root_uri": "",
                            "data_type": "",
                            "datasets": [],
                            "project_id": "live",
                        };
                        log.debug("Initialized data_project_dict:", data_project_dict);
                    }
                    if (!Array.isArray(live_indices)) live_indices = [];

                    // Check if selected_models is None/null - if so, prevent the update
                    if (selected_models === null || selected_models === undefined) {
                        log.warn("Selected models is None/null - preventing update");
                        return [buffer_data, data_project_dict, live_indices,
                               {
                                   "position": "fixed",
                                   "top": 0,
                                   "left": 0,
                                   "width": "100%",
                                   "height": "100%",
                                   "backgroundColor": "rgba(0, 0, 0, 0.7)",
                                   "zIndex": 9998,
                                   "display": "none"  // Hide the spinner when no models selected
                               },
                               false];
                    }

                    // Default spinner and transition states
                    let spinner_style = {
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0, 0, 0, 0.7)",
                        "zIndex": 9998,
                        "display": "block"
                    };
                    let transition_state = true;

                    if (message && message.data) {
                        let data = message.data;
                        log.debug("Raw message data:", data);

                        if (typeof data === "string") {
                            try {
                                data = JSON.parse(data);
                                log.debug("Parsed message data:", data);
                            } catch (e) {
                                log.error("Failed to parse message data:", e);
                                return [buffer_data, data_project_dict, live_indices, spinner_style, transition_state];
                            }
                        }

                        // Extract model information from message
                        let autoencoder_model = data.autoencoder_model;
                        let dimred_model = data.dimred_model;
                        log.debug("Message models - Autoencoder:", autoencoder_model, "Dimred:", dimred_model);

                        // Check if model names match currently selected models
                        if (selected_models !== null && selected_models !== undefined) {
                            let current_autoencoder = selected_models.autoencoder;
                            let current_dimred = selected_models.dimred;
                            log.debug("Current selected models - Autoencoder:", current_autoencoder, "Dimred:", current_dimred);

                            // Skip buffer entries from different models
                            if ((autoencoder_model && autoencoder_model !== current_autoencoder) ||
                                (dimred_model && dimred_model !== current_dimred)) {
                                log.info(`Skipping buffer entries from different models: got ${autoencoder_model}/${dimred_model}, expected ${current_autoencoder}/${current_dimred}`);

                                // Return current state without modifications when models don't match
                                return [buffer_data, data_project_dict, live_indices, spinner_style, transition_state];
                            }

                            // If we got a matching model message, hide the spinner
                            log.info("Models match - hiding spinner");
                            spinner_style = {
                                "position": "fixed",
                                "top": 0,
                                "left": 0,
                                "width": "100%",
                                "height": "100%",
                                "backgroundColor": "rgba(0, 0, 0, 0.7)",
                                "zIndex": 9998,
                                "display": "none"  // Hide the spinner
                            };
                            transition_state = false;
                        }

                        // Process the data (existing logic)
                        let new_entry = {};
                        if (data.feature_vector) {
                            log.debug("Feature vector found:", data.feature_vector);
                            new_entry["feature_vector"] = data.feature_vector;
                            new_entry["num_components"] = data.feature_vector.length;
                        } else {
                            log.debug("No feature vector in data.");
                        }

                        buffer_data = [...buffer_data, new_entry];
                        log.debug("Updated buffer_data:", buffer_data);

                        let tiled_url = data.tiled_url;
                        let index = parseInt(data.index);
                        log.debug("Tiled URI:", tiled_url, "Index:", index);

                        let url = new URL(tiled_url);
                        let path_parts = url.pathname.split('/');
                        let root_uri = tiled_url;
                        let uri = "";

                        if (path_parts.length > 1 && path_parts[path_parts.length - 1] !== '') {
                            let root_path = path_parts.slice(0, -1).join('/') + '/';
                            root_uri = url.protocol + '//' + url.host + root_path;
                            uri = path_parts[path_parts.length - 1];
                        }

                        log.debug("Root URI:", root_uri, "URI:", uri);

                        if (index >= 0) {
                            live_indices = [...live_indices, index];
                            log.debug("Updated live_indices:", live_indices);
                        }

                        let cum_size = Math.max(...live_indices) + 1;
                        log.debug("Cumulative size:", cum_size);

                        if (data_project_dict["root_uri"] !== root_uri) {
                            data_project_dict = {
                                ...data_project_dict,
                                "root_uri": root_uri,
                                "data_type": "tiled"
                            };
                            log.info("Updated data_project_dict root_uri and data_type:", data_project_dict);
                        }

                        if (data_project_dict["datasets"].length === 0) {
                            data_project_dict = {
                                ...data_project_dict,
                                "datasets": [{
                                    "uri": uri,
                                    "cumulative_data_count": cum_size
                                }]
                            };
                            log.debug("Initialized datasets in data_project_dict:", data_project_dict["datasets"]);
                        } else {
                            data_project_dict = {
                                ...data_project_dict,
                                "datasets": [
                                    ...data_project_dict["datasets"],
                                    {
                                        "uri": uri,
                                        "cumulative_data_count": cum_size
                                    }
                                ]
                            };
                            log.debug("Appended to datasets in data_project_dict:", data_project_dict["datasets"]);
                        }
                    }

                    return [buffer_data, data_project_dict, live_indices, spinner_style, transition_state];

                } catch (error) {
                    log.error("Error in clientside callback:", error);
                    return [
                        Array.isArray(buffer_data) ? buffer_data : [],
                        data_project_dict || { "root_uri": "", "data_type": "", "datasets": [], "project_id": "live" },
                        Array.isArray(live_indices) ? live_indices : [],
                        {
                            "position": "fixed",
                            "top": 0,
                            "left": 0,
                            "width": "100%",
                            "height": "100%",
                            "backgroundColor": "rgba(0, 0, 0, 0.7)",
                            "zIndex": 9998,
                            "display": "block"
                        },
                        true
                    ];
                }
            }

            // Return current state when not in live mode (n_clicks is even or null)
            return [
                buffer_data,
                data_project_dict,
                live_indices,
                {
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100%",
                    "height": "100%",
                    "backgroundColor": "rgba(0, 0, 0, 0.7)",
                    "zIndex": 9998,
                    "display": "none"
                },
                false
            ];
        }
    }
});
