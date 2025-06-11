window.dash_clientside = Object.assign({}, window.dash_clientside, {
    liveWS: {
        updateLiveData: function(message, buffer_data, n_clicks, data_project_dict, live_indices) {
            if (n_clicks !== null && n_clicks % 2 === 1) {
                try {
                    console.log("Clientside callback triggered with message:", message);
                    console.log("Current buffer_data:", buffer_data, "Type:", typeof buffer_data);

                    if (!Array.isArray(buffer_data)) buffer_data = [];
                    if (data_project_dict = {}) {
                        data_project_dict = {
                            "root_uri": "",
                            "data_type": "",
                            "datasets": [],
                            "project_id": "live",
                        };
                        console.log("Initialized data_project_dict:", data_project_dict);
                    }
                    if (!Array.isArray(live_indices)) live_indices = [];

                    if (message && message.data) {
                        let data = message.data;
                        console.log("Raw message data:", data);

                        if (typeof data === "string") {
                            try {
                                data = JSON.parse(data);
                                console.log("Parsed message data:", data);
                            } catch (e) {
                                console.error("Failed to parse message data:", e);
                                return [buffer_data, data_project_dict, live_indices];
                            }
                        }

                        let new_entry = {};
                        if (data.feature_vector) {
                            console.log("Feature vector found:", data.feature_vector);
                            new_entry["feature_vector"] = data.feature_vector;
                            new_entry["num_components"] = data.feature_vector.length;
                        } else {
                            console.log("No feature vector in data.");
                        }

                        buffer_data = [...buffer_data, new_entry];
                        console.log("Updated buffer_data:", buffer_data);

                        let tiled_url = data.tiled_url;
                        let index = parseInt(data.index);
                        console.log("Tiled URI:", tiled_url, "Index:", index);

                        let url = new URL(tiled_url);
                        let path_parts = url.pathname.split('/');
                        let root_uri = tiled_url;
                        let uri = "";

                        if (path_parts.length > 1 && path_parts[path_parts.length - 1] !== '') {
                            let root_path = path_parts.slice(0, -1).join('/') + '/';
                            root_uri = url.protocol + '//' + url.host + root_path;
                            uri = path_parts[path_parts.length - 1];
                        }

                        console.log("Root URI:", root_uri, "URI:", uri);

                        if (index >= 0) {
                            live_indices = [...live_indices, index];
                            console.log("Updated live_indices:", live_indices);
                        }

                        let cum_size = Math.max(...live_indices) + 1;
                        console.log("Cumulative size:", cum_size);

                        if (data_project_dict["root_uri"] !== root_uri) {
                            data_project_dict = {
                                ...data_project_dict,
                                "root_uri": root_uri,
                                "data_type": "tiled"
                            };
                            console.log("Updated data_project_dict root_uri and data_type:", data_project_dict);
                        }

                        if (data_project_dict["datasets"].length === 0) {
                            data_project_dict = {
                                ...data_project_dict,
                                "datasets": [{
                                    "uri": uri,
                                    "cumulative_data_count": cum_size
                                }]
                            };
                            console.log("Initialized datasets in data_project_dict:", data_project_dict["datasets"]);
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
                            console.log("Appended to datasets in data_project_dict:", data_project_dict["datasets"]);
                        }
                    }

                    return [buffer_data, data_project_dict, live_indices];

                } catch (error) {
                    console.error("Error in clientside callback:", error);
                    return [
                        Array.isArray(buffer_data) ? buffer_data : [],
                        data_project_dict || { "root_uri": "", "data_type": "", "datasets": [] },
                        Array.isArray(live_indices) ? live_indices : []
                    ];
                }
            }
            return [buffer_data, data_project_dict, live_indices];
        }
    }
});
