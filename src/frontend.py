from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from app_layout import app, LABEL_NAMES, cluster_options, latent_vector_options
from latentxp_utils import hex_to_rgba, generate_scatter_data
import ids
from latentxp_utils import generate_cluster_dropdown_options

#images = np.load("/app/work/data/Demoshapes.npz")['arr_0']
#assigned_labels = np.load("/app/work/data/DemoLabels.npy")
images = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/Demoshapes.npz")['arr_0']
assigned_labels = np.load("/Users/runbojiang/Desktop/mlex_latent_explorer/data/DemoLabels.npy")

# ------------------------------------------------
# SCATTER PLOT CALLBACKs
@app.callback(
    Output(ids.SCATTER, 'figure'),
    Input(ids.TABS, 'value'),
    Input(ids.N_COMPONENTS, 'value'),
    Input(ids.CLUSTER_DROPDOWN, 'value'),
    Input(ids.LABEL_DROPDOWN, 'value'),
    Input(ids.SCATTER_COLOR, 'value'),
    State(ids.LABELER, 'value'),
    State(ids.SCATTER, 'figure'),
    State(ids.SCATTER, 'selectedData')
)
def update_scatter_plot(selected_tab, n_components, 
                        cluster_selection, label_selection, scatter_color,
                        labeler_value, current_figure, selected_data):
    clusters = cluster_options[selected_tab]
    if n_components == '3':
        clusters = cluster_options['PCA_3d']
    cluster_names = {a: a for a in np.unique(clusters).astype(int)}
    
    latent_vectors = latent_vector_options[selected_tab]
    if n_components == '3':
        latent_vectors = latent_vector_options['PCA_3d']
   
    if selected_data is not None and len(selected_data.get('points', [])) > 0:
        selected_indices = [point['customdata'][0] for point in selected_data['points']]
    else:
        selected_indices = None

    scatter_data = generate_scatter_data(latent_vectors,
                                         n_components,
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
# TABS
@app.callback(
        Output(ids.CLUSTER_DROPDOWN, 'options'),
        Input(ids.TABS, 'value')
)
def update_cluster_dropdown(selected_tab):
    clusters = cluster_options[selected_tab]
    return generate_cluster_dropdown_options(clusters)

# -------------------------------------------------
# Dimension reduction parameters, dependes on which tab (pca or umap) is clicked
@app.callback(
        Output(ids.DR_PARAMETERS, 'children'),
        Input(ids.TABS, 'value')
)
def update_dimension_reduction_parameters(selected_tab):
    if selected_tab == 'PCA':
        return [
            html.Label('Select parameters for PCA: '),
            html.Label('Number of principal components to keep:'),
            dcc.RadioItems(id=ids.N_COMPONENTS, 
                           options=[{'label': '2', 'value': '2'},
                                    {'label': '3', 'value': '3'}],
            value='2')]
    elif selected_tab == 'UMAP':
        return [
            html.Label('Select parameters for UMAP: '),
            html.Label('Number of principal components to keep:'),
            dcc.RadioItems(id=ids.N_COMPONENTS, 
                           options=[{'label': '2', 'value': '2'},
                                    {'label': '3', 'value': '3'}], value='2'),
            html.Label('Min distance between points:'),
            html.Div(dcc.Slider(min=0.1, max=0.9, step=0.1, value=0.1, id=ids.MIN_DIST), 
                     style={'width': '50%'}),
            html.Label('Number of nearest neighbors:'),
            html.Div(dcc.Slider(min=5, max=50, step=5, value=15, id=ids.N_NEIGIBORS),
                    style={'width': '50%'})
            ]
    else:
        return None

# -------------------------------------------------
# IMAGE PANEL
@app.callback(
    Output(ids.HEATMAP, 'figure'),
    Input(ids.SCATTER, 'clickData'),
    Input(ids.SCATTER, 'selectedData'),
    Input(ids.MEAN_STD_TOGGLE, 'value'),
    State(ids.HEATMAP, 'figure')
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
    Output(ids.STATS_DIV, 'children'),
    Input(ids.TABS, 'value'),
    Input(ids.SCATTER, 'selectedData'),
    Input('assign-labels-button', 'n_clicks'),
)
def update_statistics(selected_tab, selected_data, n_clicks):
    clusters = cluster_options[selected_tab]
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
        label_int_to_str_map = {val: key for key, val in LABEL_NAMES.items()}
        labels_str = ", ".join(str(label_int_to_str_map[label]) for label in unique_labels)
        print('labels_str: ', labels_str)
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
    State(ids.LABELER, 'value'),
    State(ids.SCATTER, 'selectedData')
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
    Input(ids.SCATTER, 'relayoutData')
)
def store_scatter_axis_range(relayout_data):
    if relayout_data and ('xaxis.range[0]' in relayout_data or 'yaxis.range[0]' in relayout_data):
        return {
            'x_range': [relayout_data.get('xaxis.range[0]', None), relayout_data.get('xaxis.range[1]', None)],
            'y_range': [relayout_data.get('yaxis.range[0]', None), relayout_data.get('yaxis.range[1]', None)]
        }
    return {}

@app.callback(
    Output('label-assign-output', 'children'),
    Input('label-assign-message', 'children')
)
def update_label_assign_output(message):
    return message
    
    
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8070, )


