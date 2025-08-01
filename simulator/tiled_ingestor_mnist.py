import logging
import tempfile

import numpy as np
from tiled.client import from_uri
from torchvision import datasets, transforms

# Define how many images per label to include
LABEL_COUNTS = {
    1: 10,
    2: 3,
    3: 1,
    4: 15,
    5: 1,
    6: 5,
    7: 7,
    8: 2,
    9: 4,
    0: 6,
}

LABEL_NAMES = {
    1: "ones",
    2: "twos",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    0: "zero",
}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: (%(name)s)  %(message)s "
)
logger.setLevel(logging.INFO)


def ingest_mnist_to_tiled(tiled_uri, api_key: str = None) -> bool:
    """Ingest MNIST dataset into Tiled under /mnist/ with subsets for each digit."""

    try:
        # Connect to Tiled and create container
        client = from_uri(tiled_uri, api_key=api_key)

        if "mnist" not in client:
            client.create_container(
                "mnist", metadata={"purpose": "MNIST digit subsets"}
            )
        else:
            logger.info(
                "⚠️ Container 'mnist' already exists, no data has been ingested."
            )
            return True

        mnist_container = client["mnist"]

        # Load MNIST dataset
        with tempfile.TemporaryDirectory() as tmp_dir:
            mnist_dataset = datasets.MNIST(
                root=tmp_dir, train=True, download=True, transform=transforms.ToTensor()
            )
        logger.debug(f"📊 Loaded MNIST dataset with {len(mnist_dataset)} images")

        # Group images by label
        images_by_label = {i: [] for i in range(10)}
        for img, label in mnist_dataset:
            images_by_label[label].append(img.squeeze().numpy())  # 28x28

        # Write arrays to Tiled
        for digit, count in LABEL_COUNTS.items():
            label_name = LABEL_NAMES[digit]
            selected_images = images_by_label[digit][:count]
            selected_array = (
                np.stack(selected_images) if count > 1 else selected_images[0]
            )
            logger.debug(
                f"Writing {len(selected_images)} images for label {label_name} (digit {digit})"
            )
            array_shape = selected_array.shape
            logger.debug(
                f"Shape of selected array: {array_shape if isinstance(selected_array, np.ndarray) else 'single image'}"
            )

            mnist_container.write_array(
                key=label_name,
                array=selected_array,
                metadata={"label": digit, "count": count},
            )

        logger.info("✅ MNIST subset successfully uploaded to Tiled")
        return True

    except Exception as e:
        import traceback

        logger.debug(traceback.format_exc())
        logger.info(f"❌ Error: {e}")
        return False
