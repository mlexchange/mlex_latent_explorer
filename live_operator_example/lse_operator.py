import os

import torch
import torchvision.transforms as transforms
from joblib import load
from model_simple_auto import CNNAutoencoder
from tiled.client import from_uri
from tiled_utils import write_results

DATA_TILED_URI = os.getenv("DATA_TILED_URI", "")
DATA_TILED_KEY = os.getenv("DATA_TILED_KEY", None)
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", None)

MODEL_CHECKPOINT_DIR = os.getenv("MODEL_CHECKPOINT_DIR", "")
DIM_REDUCTION_MODEL_DIR = os.getenv("DIM_REDUCTION_MODEL_DIR", "")

if __name__ == "__main__":
    model = CNNAutoencoder()
    checkpoint = torch.load(MODEL_CHECKPOINT_DIR)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    dim_reduction_model = load(DIM_REDUCTION_MODEL_DIR)

    # Check for CUDA else use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model = model.to(device)

    # Define the transformation to apply to the data
    transform = transforms.Compose(
        [
            transforms.Resize(
                (128, 128)
            ),  # Resize to smaller dimensions to save memory
            transforms.ToTensor(),  # Convert image to PyTorch tensor (0-1 range)
            transforms.Normalize(
                (0.0,), (1.0,)
            ),  # Normalize tensor to have mean 0 and std 1
        ]
    )

    # Set base tiled client
    data_client = from_uri(DATA_TILED_URI, api_key=DATA_TILED_KEY)
    write_client = from_uri(RESULTS_TILED_URI, api_key=RESULTS_TILED_API_KEY)

    for i in range(1000):  # TODO: Change to long running loop that detects new data
        # Assuming we get a "relative" tiled_uri (e.g. container): sub_uri
        datapoint = data_client[indx]  # noqa: F821
        tensor = transform(datapoint)  # Add batch and channel dimensions
        f_vec_nn = model.module.encoder(tensor)

        # 2. Dimension Reduction (PCA)
        f_vec = dim_reduction_model.transform(f_vec_nn)

        # 3. Save results to tiled with metadata
        write_results(
            write_client,
            f_vec,
            io_parameters,  # noqa: F821
            latent_vectors_path,  # noqa: F821
            metadata=None,  # noqa: F821
        )
