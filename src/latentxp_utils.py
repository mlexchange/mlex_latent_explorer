import plotly.graph_objects as go
import numpy as np
import colorsys
from copy import deepcopy
import requests
import os
from PIL import Image

kmeans_kwargs = {"gui_parameters": [{"type": "dropdown", "name": "ncluster-dropdown-menu", "title": "Number of clusters", "param_key": "n_clusters",
                                     "options": [{"label": i, "value": i} for i in range(1, 21)], 
                                     "value":8},
                                     ]
                }
dbscan_kwargs = {"gui_parameters": [
                                    {"type": "dropdown", "name": "eps-dropdown", "title": "The maximum distance between two samples for one to be considered as in the neighborhood of the other", "param_key": "eps",
                                     "options": [{"label": i/10, "value": i/10} for i in range(1, 31)], 
                                     "value":0.5},
                                    {"type": "dropdown", "name": "minsample-dropdown", "title": "The number of samples in a neighborhood for a point to be considered as a core point", "param_key": "min_samples",
                                     "options": [{"label": i, "value": i} for i in range(1, 31)], 
                                     "value":5},
                                    ]
                }
hdbscan_kwargs = {"gui_parameters": [
                                    {"type": "dropdown", "name": "mincluster-size", "title": "The minimum number of samples in a group for that group to be considered a cluster", "param_key": "min_cluster_size",
                                     "options": [{"label": i, "value": i} for i in range(2, 31)], 
                                     "value":5},
                                    ]
                }

def check_if_path_exist(path):
    if os.path.exists(path):
        print(f"The path '{path}' exists.")
    else:
        print(f"The path '{path}' does not exist.")

def get_job(user, mlex_app):
    url = 'http://job-service:8080/api/v0/jobs?'
    if user:
        url += ('&user=' + user)
    if mlex_app:
        url += ('&mlex_app=' + mlex_app)
    
    response = requests.get(url).json()
    return response

def get_content(uid: str):
    url = 'http://content-api:8000/api/v0/contents/{}/content'.format(uid)  # current host, could be inside the docker
    response = requests.get(url).json()
    return response

def remove_key_from_dict_list(data, key):
    new_data = []
    for item in data:
        if key in item:
            # print("key in item")
            new_item = deepcopy(item)
            new_item.pop(key)
            new_data.append(new_item)
        else:
            new_data.append(item)
    
    return new_data


def hex_to_rgba(hex_color, alpha=1.0):
    """
    Converts a hex color string to an RGBA color string.

    Parameters:
    hex_color (str): The color to convert, in hex format.
    alpha (float, optional): The alpha (opacity) value of the color. Default is 1.0.

    Returns:
    str: The RGBA color string.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def generate_colors(num_colors):
    """
    Generates a list of color codes.

    Parameters:
    num_colors (int): The number of color codes to generate.

    Returns:
    list: A list of color codes.
    """
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5
        saturation = 0.8
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    return colors


def generate_scattergl_plot(x_coords, y_coords, labels, label_to_string_map, show_legend=False, custom_indices=None):
    """
    Generates a two dimensional Scattergl plot.

    Parameters:
    x_coords (list): The x-coordinates of the points.
    y_coords (list): The y-coordinates of the points.
    labels (list): The labels of the points.
    label_to_string_map (dict): A mapping from labels to strings.
    show_legend (bool, optional): Whether to show a legend. Default is False.
    custom_indices (list, optional): Custom indices for the points. Default is None.

    Returns:
    go.Figure: The generated Scattergl plot.
    """
    # Create a set of unique labels
    unique_labels = set(labels)

    # Create a trace for each unique label
    traces = []
    for label in unique_labels:
        # Find the indices of the points with the current label
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]

        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        traces.append(
            go.Scattergl(
                x=trace_x,
                y=trace_y,
                customdata=np.array(trace_custom_indices).reshape(-1, 1),
                mode='markers',
                name=str(label_to_string_map[label])
            )
        )

    # Create the plot with the scatter plot traces
    fig = go.Figure(data=traces)
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(255, 255, 255, 0.9)',
                orientation='h'
            )
        )
    return fig


def generate_scatter3d_plot(x_coords, y_coords, z_coords, labels, label_to_string_map, show_legend=False, custom_indices=None):
    """
    Generates a three-dimensional Scatter3d plot.

    Parameters:
    x_coords (list): The x-coordinates of the points.
    y_coords (list): The y-coordinates of the points.
    z_coords (list): The z-coordinates of the points.
    labels (list): The labels of the points.
    label_to_string_map (dict): A mapping from labels to strings.
    show_legend (bool, optional): Whether to show a legend. Default is False.
    custom_indices (list, optional): Custom indices for the points. Default is None.

    Returns:
    go.Figure: The generated Scatter3d plot.
    """
    # Create a set of unique labels
    unique_labels = set(labels)

    # Create a trace for each unique label
    traces = []
    for label in unique_labels:
        # Find the indices of the points with the current label
        trace_indices = [i for i, l in enumerate(labels) if l == label]
        trace_x = [x_coords[i] for i in trace_indices]
        trace_y = [y_coords[i] for i in trace_indices]
        trace_z = [z_coords[i] for i in trace_indices]

        if custom_indices is not None:
            trace_custom_indices = [custom_indices[i] for i in trace_indices]
        else:
            trace_custom_indices = trace_indices

        traces.append(
            go.Scatter3d(
                x=trace_x,
                y=trace_y,
                z=trace_z,
                customdata=np.array(trace_custom_indices).reshape(-1, 1),
                mode='markers',
                name=str(label_to_string_map[label]),
                marker=dict(size=3),
            )
        )

    # Create the plot with the Scatter3d traces
    fig = go.Figure(data=traces)
    if show_legend:
        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(255, 255, 255, 0.9)',
                orientation='h'
            )
        )
    return fig


def generate_scatter_data(latent_vectors,
                          n_components,
                          cluster_selection=-1, #"All"
                          clusters=None,
                          cluster_names=None,
                          label_selection=-2, #"All"
                          labels=None,
                          label_names=None,
                          color_by=None,
                          ):
    """
    Generate data for a plot according to the provided selection options:
    1. all clusters & all labels
    2. all clusters and selected labels
    3. all labels and selected clusters
    4. selected clusters and selected labels

    Parameters:
    latent_vectors (numpy.ndarray, Nx2, floats): [Description]
    n_components: number principal components
    cluster_selection (int): The cluster w want to select. Defaults to -1: all clusters
    clusters (numpy.ndarray, N, ints optional): The cluster number for each data point
    cluster_names (dict, optional): [Description]. A dictionary with cluster names
    label_selection (str, optional): Which label to select. Defaults to -2: all labels. -1 mean Unlabeled
    labels (numpy.ndarray, N, int, optional): The current labels Defaults to None.
    label_names (dict, optional): A dictionary that relates label number to name.
    color_by (str, optional): Determines if we color by label or cluster. Defaults to None.

    Returns:
    plotly.scattergl: A plot as specified.
    """
    # case:
    #  all data: cluster_selection =-1, label_selection=-2
    #  all clusters, selected labels
    #  all labels, selected clusters

    vals_names = {}
    if color_by == 'cluster':
        vals = clusters
        vals_names = cluster_names
    if color_by == 'label':
        vals = labels
        if label_names is not None:
            vals_names = {value: key for key, value in label_names.items()}
        vals_names[-1] = "Unlabeled"
        
    if (cluster_selection == -1) & (label_selection == -2):
        if n_components == 2:
            scatter_data = generate_scattergl_plot(latent_vectors[:, 0],
                                                latent_vectors[:, 1],
                                                vals,
                                                vals_names)
            return scatter_data
        else:
            scatter_data = generate_scatter3d_plot(latent_vectors[:, 0],
                                                latent_vectors[:, 1],
                                                latent_vectors[:, 2],
                                                vals,
                                                vals_names)
            return scatter_data

    selected_indices = None
    clusters = np.array(clusters)
    labels = np.array(labels)
    if (cluster_selection == -1) & (label_selection != -2):  # all clusters
        if label_selection != -1:
            label_selection = label_names[label_selection]
        selected_indices = np.where(labels == label_selection)[0]

    if (label_selection == -2) & (cluster_selection > -1):  # all clusters
        selected_indices = np.where(clusters == cluster_selection)[0]

    if (label_selection != -2) & (cluster_selection > -1):
        if label_selection != -1:
            selected_labels = label_names[label_selection]
            selected_indices = np.where((clusters == cluster_selection) & (labels == selected_labels))[0]
        else:
            selected_indices = np.where((clusters == cluster_selection))[0]

    vals = np.array(vals)
    if n_components == 2: 
        scatter_data = generate_scattergl_plot(latent_vectors[selected_indices, 0],
                                            latent_vectors[selected_indices, 1],
                                            vals[selected_indices],
                                            vals_names,
                                            custom_indices=selected_indices)
    elif n_components == 3:
        scatter_data = generate_scatter3d_plot(latent_vectors[selected_indices, 0],
                                            latent_vectors[selected_indices, 1],
                                            latent_vectors[selected_indices, 2],
                                            vals[selected_indices],
                                            vals_names,
                                            custom_indices=selected_indices)
    return scatter_data


def compute_mean_std_images(selected_indices, images):
    """
    Computes the mean and standard deviation of a selection of images.

    Parameters:
    selected_indices (list): The indices of the selected images.
    images (numpy.ndarray): The array of all images.

    Returns:
    tuple: A tuple containing the mean image and the standard deviation image.
    """
    selected_images = images[selected_indices]
    mean_image = np.mean(selected_images, axis=0)
    std_image = np.std(selected_images, axis=0)
    return mean_image, std_image

def get_trained_models_list(user, app):
    '''
    This function queries the MLCoach or DataClinic results
    Args:
        user:               Username
        app:                Tab option (MLCoach vs Data Clinic)
        similarity:         [Bool] Retrieve f_vec vs probabilities
    Returns:
        trained_models:     List of options
    '''
    MLEX_COMPUTE_URL = "http://job-service:8080/api/v0"
    filename = '/f_vectors.parquet'
    model_list = requests.get(f'{MLEX_COMPUTE_URL}/jobs?&user={user}&mlex_app={app}').json()
    # print(model_list, flush=True)
    trained_models = []
    for model in model_list:
        if model['job_kwargs']['kwargs']['job_type']=='prediction_model':
            cmd = model['job_kwargs']['cmd'].split(' ')
            indx = cmd.index('-o')
            out_path = cmd[indx+1]
            if os.path.exists(out_path+filename):  # check if the file exists
                if model['description']:
                    trained_models.append({'label': app+': '+model['description'],
                                           'value': out_path+filename})
                else:
                    trained_models.append({'label': app+': '+model['job_kwargs']['kwargs']['job_type'],
                                           'value': out_path+filename})
    trained_models.reverse()
    return trained_models


def load_images_from_directory(directory_path, indices):
    image_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            file_path = os.path.join(directory_path, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                image_data.append(img_array)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    image_data = np.array(image_data)
    return image_data

def load_images_by_indices(directory_path, indices):
    image_data = []
    filenames = [filename for filename in sorted(os.listdir(directory_path)) if filename.lower().endswith(('.png', '.jpg'))]
    for index in indices:
        if index in range(len(filenames)):
            filename = filenames[index]
            file_path = os.path.join(directory_path, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                image_data.append(img_array)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    image_data = np.array(image_data)
    return image_data
