// Set this in your HTML or another script to configure log level
// window.DASH_LOG_LEVEL = 3; // INFO level by default

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    liveWS: {
        updateLiveData: function(message, buffer_data, n_clicks, data_project_dict, live_indices, selected_models) {
            // Initialize log level if not set
            window.DASH_LOG_LEVEL = 4; // window.DASH_LOG_LEVEL || 3;

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
                    log.debug("Current data_project_dict:", data_project_dict);

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

                        // Compare full "name:version" identifiers
                        if (selected_models !== null && selected_models !== undefined) {
                            // Construct full identifiers from selected_models
                            let current_autoencoder_id = selected_models.autoencoder;
                            let current_dimred_id = selected_models.dimred;
                            
                            // Add version if available
                            if (selected_models.autoencoder_version) {
                                current_autoencoder_id = `${selected_models.autoencoder}:${selected_models.autoencoder_version}`;
                            }
                            if (selected_models.dimred_version) {
                                current_dimred_id = `${selected_models.dimred}:${selected_models.dimred_version}`;
                            }
                            
                            log.debug("Current selected model IDs - Autoencoder:", current_autoencoder_id, "Dimred:", current_dimred_id);

                            // Skip buffer entries from different model versions
                            if ((autoencoder_model && autoencoder_model !== current_autoencoder_id) ||
                                (dimred_model && dimred_model !== current_dimred_id)) {
                                log.info(`Skipping buffer entries from different model versions: got ${autoencoder_model}/${dimred_model}, expected ${current_autoencoder_id}/${current_dimred_id}`);

                                // Return current state without modifications when models don't match
                                return [buffer_data, data_project_dict, live_indices, spinner_style, transition_state];
                            }

                            // If we got a matching model message, hide the spinner
                            log.info("Model versions match - hiding spinner");
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

                        // Process the data
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

                        let url = new URL(tiled_url);
                        const path_parts = url.pathname.split('/').filter(p => p !== '');
                        let root_uri = url.origin;
                        let uri = "";

                        // Find base URI (e.g., api/v1/metadata or api/v1/array/full)
                        const apiIndex = path_parts.findIndex((p, i) =>
                            p === 'api' &&
                            path_parts[i + 1] === 'v1' &&
                            (
                                path_parts[i + 2] === 'metadata' ||
                                (path_parts[i + 2] === 'array' && ['full', 'block'].includes(path_parts[i + 3]))
                            )
                        );

                        if (apiIndex !== -1) {
                            let base_root_parts;
                            if (path_parts[apiIndex + 2] === 'metadata') {
                                base_root_parts = path_parts.slice(0, apiIndex + 3);
                            } else {
                                base_root_parts = path_parts.slice(0, apiIndex + 4);
                            }

                            root_uri = `${url.protocol}//${url.host}/${base_root_parts.join('/')}`;
                            uri = decodeURIComponent(path_parts.slice(base_root_parts.length).join('/'));
                        } else {
                            console.warn("Unexpected Tiled URL format:", tiled_url);
                        }

                        // Check for slice index in the query string
                        const params = new URLSearchParams(url.search);
                        const sliceParam = params.get('slice');
                        let index = 0;

                        if (sliceParam) {
                            // Expecting format like "0,0,:,:" or "0:1,0:1679,0:1475"
                            const sliceParts = sliceParam.split(','); 
                            const firstSliceParts = sliceParts[0];
                            
                            if (firstSliceParts.includes(':')) {
                                // New format: extract the first number before the colon (e.g., "0" from "0:1")
                                const startIndex = parseInt(firstSliceParts.split(':')[0], 10);
                                if (!isNaN(startIndex)) {
                                    index = startIndex;
                                }
                            } else {
                                // Old format: directly parse the first part as an integer
                                const parsedIndex = parseInt(firstSliceParts, 10);
                                if (!isNaN(parsedIndex)) {
                                    index = parsedIndex;
                                }
                            }
                        } else {
                            // No slice param, assume single data point
                            index = 0;
                            console.warn(`No 'slice' parameter found in Tiled URL: ${tiled_url}`);
                        }

                        log.debug("Root URI:", root_uri, "URI:", uri, "Index:", index);
                        
                        // Change root_uri from /api/v1/array/full to /api/v1/metadata
                        if (root_uri.includes('/api/v1/array/full')) {
                            root_uri = root_uri.replace('/api/v1/array/full', '/api/v1/metadata');
                            log.debug("Modified Root URI:", root_uri);
                        }

                        if (index < 0) {
                            log.warn("Received negative index; skipping update");
                            return [
                                window.dash_clientside.no_update,
                                window.dash_clientside.no_update,
                                window.dash_clientside.no_update,
                                window.dash_clientside.no_update,
                                window.dash_clientside.no_update
                            ];
                        }

                        if (data_project_dict["root_uri"] !== root_uri) {
                            data_project_dict = {
                                ...data_project_dict,
                                "root_uri": root_uri,
                                "data_type": "tiled"
                            };
                            log.info("Updated data_project_dict root_uri and data_type:", data_project_dict);
                        }

                        let currentCumulative;

                        if (data_project_dict["datasets"].length === 0) {
                            log.debug("Initializing datasets with raw index:", index, "URI:", uri);

                            currentCumulative = index; // Since cumulative_data_count = index + 1

                            data_project_dict = {
                                ...data_project_dict,
                                "datasets": [{
                                    "uri": uri,
                                    "cumulative_data_count": currentCumulative + 1
                                }]
                            };

                            log.debug("Initialized data_project_dict:", data_project_dict, "with cumulative count:", currentCumulative + 1, "and index:", index);

                        } else {
                            const existingDataset = data_project_dict["datasets"].find(d => d.uri === uri);

                            if (existingDataset) {
                                // Update existing dataset cumulative count
                                const datasetIndex = data_project_dict["datasets"].findIndex(d => d.uri === uri);
                                const prevCumulative = datasetIndex > 0
                                    ? data_project_dict["datasets"][datasetIndex - 1].cumulative_data_count
                                    : 0;

                                currentCumulative = prevCumulative + index;

                                const newCumulative = Math.max(existingDataset.cumulative_data_count, currentCumulative + 1);

                                // Only proceed if we're actually updating the cumulative_data_count
                                if (newCumulative !== existingDataset.cumulative_data_count) {
                                    existingDataset.cumulative_data_count = newCumulative;

                                    // Update cumulative_data_count for all subsequent datasets
                                    for (let i = datasetIndex + 1; i < data_project_dict["datasets"].length; i++) {
                                        const previousDataset = data_project_dict["datasets"][i - 1];
                                        data_project_dict["datasets"][i].cumulative_data_count =
                                            previousDataset.cumulative_data_count + 1;
                                    }
                                }


                                log.debug("Updated existing dataset cumulative count:", data_project_dict);

                            } else {
                                // Append new dataset with cumulative count
                                const lastCumulative = data_project_dict["datasets"].at(-1).cumulative_data_count;

                                currentCumulative = lastCumulative + index;

                                data_project_dict["datasets"].push({
                                    "uri": uri,
                                    "cumulative_data_count": currentCumulative + 1
                                });

                                log.debug("Appended new dataset with cumulative count:", currentCumulative + 1, "data_project_dict:", data_project_dict);
                            }
                        }

                        // Track the cumulative index of the current slice
                        live_indices = [...live_indices, currentCumulative];
                        log.debug("Updated live_indices:", live_indices);
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
                            "display": "none"
                        },
                        false
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