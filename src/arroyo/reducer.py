import glob
import importlib
import logging
import os
import sys
from pathlib import Path

from arroyosas.schemas import RawFrameEvent
import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)


# message = {
#     "tiled_uri": DATA_TILED_URI,
#     "index": index,
#     "feature_vector": latent_vector.tolist(),
# }


class LatentSpaceReducer:
    """
    Responsible for taking an image, encoding it into a
    latent space, and saving the latent space to a Tiled dataset.
    The encoding is down two ways. First, it's through
    a CNN autoencoder.
    Second, it's through a dimension reduction
    agorithm to a 2D space. The results are saved to a Tiled dataset.
    """

    def __init__(
        self, current_latent_space: str, current_dim_reduction: str, models_config
    ):
        self.current_latent_space = current_latent_space
        self.current_dim_reduction = current_dim_reduction
        self.models_config = models_config
        # Check for CUDA else use CPU
        # needs to be members of the reducer class
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        self.device = device
        self.model_cache = {}

        # Load the models now
        self.current_torch_model = self.get_ls_model()
        self.curent_dim_reduction_model = self.get_dim_reduction_model()
        self.current_transform = self.get_transform()

    def reduce(self, message: RawFrameEvent) -> np.ndarray:
        # 1. Encode the image into a latent space. For now we assume
        pil = Image.fromarray(message.image.array.astype(np.float32))
        tensor = self.current_transform(pil)
        logger.debug("Encoding image into latent space")
        ls_is_on_gpu = False
        if (
            self.device == torch.device("cuda")
            and self.current_latent_space["config"].type == "torch"
        ):
            ls_is_on_gpu = True

        tensor = tensor.unsqueeze(0).to(self.device)
        latent_space, _ = self.current_torch_model["model"].encoder(tensor)

        # 2. Reduce the latent space to a 2D space
        logger.debug("Reducing latent space")
        if self.curent_dim_reduction_model["config"].type == "joblib":
            if ls_is_on_gpu:
                latent_space = latent_space.cpu().detach()
            else:
                latent_space = latent_space.detach()

            latent_space = latent_space.view(latent_space.size(0), -1)
            f_vec = self.curent_dim_reduction_model["model"].transform(latent_space.numpy())
        else:  # it's torch
            f_vec = self.current_torch_model.encoder(latent_space)
        logger.debug(f"Reduced latent space to {f_vec.shape}")
        return f_vec

    def get_ls_model(self):
        ls_model_name = self.current_latent_space
        logger.info(f"Loading Latent Space model {self.current_latent_space}")
        model = self.get_model(ls_model_name)
        logger.info("Latent Space Model Loaded")
        return model

    def get_dim_reduction_model(self):
        dim_reduction_name = self.current_dim_reduction
        logger.info(f"Loading Dimensionality Reduction Model {dim_reduction_name}")
        model = self.get_model(dim_reduction_name)
        logger.info("Dimensionality Reduction model loaded")
        return model

    def get_model(self, name: str):
        # check for model in configured models
        current_model_config = None
        for model_config in self.models_config:
            if model_config.name == name:
                current_model_config = model_config
                break
        if current_model_config is None:
            raise ValueError(f"Current model {name} is not found in models")

        # check if model is in cache. If not, load it and add to cache
        if current_model_config.name not in self.model_cache:
            loaded_model = self.load_model(model_config)
            self.model_cache[current_model_config.name] = {
                "model": loaded_model,
                "config": model_config,
            }
        return self.model_cache[current_model_config.name]

    def load_model(self, model_config):
        if model_config.type == "torch":
            model = self.load_torch_model(model_config)
            # model = model().to(self.device)
            model = model(latent_dim=512).to(self.device)
            return model
        return joblib.load(model_config.file)

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Resize to smaller dimensions to save memory
                transforms.ToTensor(),  # Convert image to PyTorch tensor (0-1 range)
                transforms.Normalize(
                    (0.0,), (1.0,)
                ),  # Normalize tensor to have mean 0 and std 1
            ]
        )
    
    
    def get_transform_old(self):
        return transforms.Compose(
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

    def import_module_from_path(self, module_name, file_path):
        file_path = Path(file_path).resolve()  # Get absolute path
        module_name = file_path.stem  # Get filename without extension as module name
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_torch_model(self, model_config: dict) -> torch.nn.Module:
        module = self.import_module_from_path(
            model_config.python_class, model_config.python_file
        )
        model = getattr(module, model_config.python_class)

        # Load the .npz file
        npz_data = np.load(model_config["state_dict"], allow_pickle=True)

        # Convert NumPy arrays to PyTorch tensors
        state_dict = {key: torch.tensor(value) for key, value in npz_data.items()}

        # Load the state dictionary into the model
        model(latent_dim=512).load_state_dict(state_dict)
        return model

    def import_torch_models(self):
        torch_models = []
        model_dir = self.settings.models.model_dir
        py_files = glob.glob(os.path.join(model_dir, "**/*.py"), recursive=True)

        for py_file in py_files:
            module_name = os.path.splitext(os.path.relpath(py_file, model_dir))[
                0
            ].replace(os.sep, ".")
            module = importlib.import_module(module_name)
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                    logger.info(f"Found torch model: {name} in {module_name}")
                    torch_models.append(obj)
        return torch_models

    @classmethod
    def from_settings(cls, settings) -> "LatentSpaceReducer":
        reducer = cls(
            settings.current_latent_space,
            settings.current_dim_reduction,
            settings.models,
        )

        return reducer


# if __name__ == "__main__":
#     reducer = LatentSpaceReducer.from_settings(default_settings.lse)
#     reducer.reduce()
