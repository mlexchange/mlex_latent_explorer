import os

import load_dotenv
import numpy as np
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure
from torchvision import datasets, transforms

# Load .env for API key
load_dotenv.load_dotenv()
tiled_api_key = os.getenv("RESULTS_TILED_API_KEY")

# Connect to root of Tiled server
client = from_uri("http://tiled:8000/api/v1", api_key=tiled_api_key)

# Create a container at /mnist if it doesn't exist
if "mnist" not in client:
    client.create_container("mnist", metadata={"purpose": "MNIST digit subsets"})

mnist_container = client["mnist"]

# Load MNIST dataset
mnist_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

# Group images by label
images_by_label = {i: [] for i in range(10)}
for img, label in mnist_dataset:
    images_by_label[label].append(img.squeeze().numpy())  # 28x28

# Define how many images per label to include
label_mapping = {
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

label_names = {
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

# Write digit subsets into the mnist container
for digit, count in label_mapping.items():
    label_name = label_names[digit]
    selected_images = images_by_label[digit][:count]
    selected_array = np.stack(selected_images) if count > 1 else selected_images[0]
    structure = ArrayStructure.from_array(selected_array)

    mnist_container.write_array(
        key=label_name, array=selected_array, metadata={"label": digit, "count": count}
    )


print("✅ MNIST subset successfully uploaded to Tiled under /mnist/")
