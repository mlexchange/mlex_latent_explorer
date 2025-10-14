import numpy as np
from arroyopy.schemas import Event
from pydantic import BaseModel, field_serializer, field_validator


class SerializableNumpyArrayModel(BaseModel):
    """
    Custom Pydantic model for serializing NumPy arrays.
    """

    array: np.ndarray

    @field_serializer("array")
    def serialize_array(self, value: np.ndarray):
        """Convert NumPy array to a dictionary with bytes and dtype"""
        return {
            "data": value.tobytes(),
            "dtype": str(value.dtype.name),
            "shape": value.shape,
        }

    @field_validator("array", mode="before")
    @classmethod
    def deserialize_array(cls, value):
        """Convert bytes back to NumPy array"""
        if isinstance(value, dict) and "data" in value:
            return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(
                value["shape"]
            )
        return value

    class Config:
        arbitrary_types_allowed = True

class LatentSpaceEvent(Event):
    tiled_url: str
    feature_vector: list[float]
    index: int
    autoencoder_model: str = None  # Add autoencoder model name
    dimred_model: str = None       # Add dimension reduction model name
    experiment_name: str = None    # NEW: Add experiment name
    timestamp: float = None        # Add timestamp for when processing began
    total_processing_time: float = None  # Total time to process the frame
    autoencoder_time: float = None  # Time spent in autoencoder processing
    dimred_time: float = None      # Time spent in dimension reduction processing