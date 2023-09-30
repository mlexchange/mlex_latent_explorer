from dash import html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from app_layout import app, LABEL_NAMES, clusters, latent_vectors
from latentxp_utils import hex_to_rgba, generate_colors, generate_scattergl_plot, generate_scatter_data, compute_mean_std_images


#images = np.load("/app/work/data/Demoshapes.npz")['arr_0']
#assigned_labels = np.load("/app/work/data/DemoLabels.npy")
images = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/Demoshapes.npz")['arr_0']
assigned_labels = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/DemoLabels.npy")
cluster_names = {a: a for a in np.unique(clusters).astype(int)}

# ------------------------------------------------
# SCATTER PLOT CALLBACKs
@app.callback(
    Output('scatter-b', 'figure'),
    Input('cluster-dropdown', 'value'),
    Input('label-dropdown', 'value'),
    Input('scatter-color', 'value'),
    State('labeler', 'value'),
    State('scatter-b', 'figure'),
    State('scatter-b', 'selectedData')
)
def update_scatter_plot(cluster_selection, label_selection, scatter_color, labeler_value, current_figure,
                        selected_data):
    if selected_data is not None and len(selected_data.get('points', [])) > 0:
        selected_indices = [point['customdata'][0] for point in selected_data['points']]
    else:
        selected_indices = None

    scatter_data = generate_scatter_data(latent_vectors,
                                         cluster_selection,
                                         clusters,
                                         cluster_names,
                                         label_selection,
                                         assigned_labels,
                                         LABEL_NAMES,
                                         scatter_color)

    fig = go.Figure(scatter_data)
    fig.update_layout(legend=dict(tracegroupgap=20))
    # print(labeler_value)

    if current_figure and 'xaxis' in current_figure['layout'] and 'yaxis' in current_figure[
        'layout'] and 'autorange' in current_figure['layout']['xaxis'] and current_figure['layout']['xaxis'][
        'autorange'] is False:
        # Update the axis range with current figure's values if available and if autorange is False
        fig.update_xaxes(range=current_figure['layout']['xaxis']['range'])
        fig.update_yaxes(range=current_figure['layout']['yaxis']['range'])
    else:
        # If it's the initial figure or autorange is True, set autorange to True to fit all points in view
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

    if selected_indices is not None:
        # Use the selected indices to highlight the selected points in the updated figure
        for trace in fig.data:
            if trace.marker.color is not None:
                trace.marker.color = [hex_to_rgba('grey', 0.3) if i not in selected_indices else 'red' for i in
                                      range(len(trace.marker.color))]
    return fig

# -------------------------------------------------
# IMAGE PANEL
@app.callback(
    Output('heatmap-a', 'figure'),
    Input('scatter-b', 'clickData'),
    Input('scatter-b', 'selectedData'),
    Input('mean-std-toggle', 'value'),
    State('heatmap-a', 'figure')
)
def update_panel_a(click_data, selected_data, display_option, current_figure):
    if selected_data is not None and len(selected_data['points']) > 0:
        selected_indices = [point['customdata'][0] for point in
                            selected_data['points']]  # Access customdata for the original indices
        selected_images = images[selected_indices]
        if display_option == 'mean':
            heatmap_data = go.Heatmap(z=np.mean(selected_images, axis=0))
        elif display_option == 'sigma':
            heatmap_data = go.Heatmap(z=np.std(selected_images, axis=0))
    elif click_data is not None and len(click_data['points']) > 0:
        selected_index = click_data['points'][0]['customdata'][0]  # click_data['points'][0]['pointIndex']
        heatmap_data = go.Heatmap(z=images[selected_index])
    else:
        heatmap_data = go.Heatmap()

    # Determine the aspect ratio based on the shape of the heatmap_data's z-values
    aspect_x = 1
    aspect_y = 1
    if heatmap_data['z'] is not None:
        if heatmap_data['z'].size > 0:
            aspect_y, aspect_x = np.shape(heatmap_data['z'])

    return go.Figure(
        data=heatmap_data,
        layout=dict(
            autosize=True,
            yaxis=dict(scaleanchor="x", scaleratio=aspect_y / aspect_x),
        )
    )

# -------------------------------------------------
# DISPLAY SELECTION STATISTICS
@app.callback(
    Output('stats-div', 'children'),
    Input('scatter-b', 'selectedData'),
    Input('assign-labels-button', 'n_clicks'),
)
def update_statistics(selected_data, n_clicks):
    if selected_data is not None and len(selected_data['points']) > 0:
        selected_indices = [point['customdata'][0] for point in
                            selected_data['points']]  # Access customdata for the original indices
        selected_clusters = clusters[selected_indices]
        selected_labels = assigned_labels[selected_indices]

        num_images = len(selected_indices)
        unique_clusters = np.unique(selected_clusters)
        unique_labels = np.unique(selected_labels)

        # Format the clusters and labels as comma-separated strings
        clusters_str = ", ".join(str(cluster) for cluster in unique_clusters)
        labels_str = ", ".join(str(LABEL_NAMES[label]) for label in unique_labels if label in LABEL_NAMES)
    else:
        num_images = 0
        clusters_str = "N/A"
        labels_str = "N/A"

    return [
        html.P(f"Number of images selected: {num_images}"),
        html.P(f"Clusters represented: {clusters_str}"),
        html.P(f"Labels represented: {labels_str}"),
    ]

@app.callback(
    Output("scatter-update-trigger", "children"),
    Input("assign-labels-button", "n_clicks"),
    State('labeler', 'value'),
    State('scatter-b', 'selectedData')
)
def trigger_scatter_update(n_clicks, labeler_value, selected_data):
    if n_clicks is not None:
        if n_clicks > 0:
            if selected_data is not None and len(selected_data['points']) > 0:
                selected_indices = [point['customdata'][0] for point in selected_data['points']]
                for idx in selected_indices:
                    if labeler_value != -1:
                        assigned_labels[idx] = LABEL_NAMES[labeler_value]
                    else:
                        assigned_labels[idx] = -1

            return n_clicks
        else:
            return n_clicks

    else:
        return n_clicks

    return n_clicks

@app.callback(
    Output('scatter-axis-range', 'data'),
    Input('scatter-b', 'relayoutData')
)
def store_scatter_axis_range(relayout_data):
    if relayout_data and ('xaxis.range[0]' in relayout_data or 'yaxis.range[0]' in relayout_data):
        return {
            'x_range': [relayout_data.get('xaxis.range[0]', None), relayout_data.get('xaxis.range[1]', None)],
            'y_range': [relayout_data.get('yaxis.range[0]', None), relayout_data.get('yaxis.range[1]', None)]
        }
    return {}

# @app.callback(
#     Output('label-assign-output', 'children'),
#     Input('label-assign-message', 'children')
# )
# def update_label_assign_output(message):
#     return message
    
    
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8070, )


