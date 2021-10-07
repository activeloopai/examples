import hub
import numpy as np
from PIL import Image


def upload(uri: str):
    """Upload some numpy data!"""

    ds = hub.empty(uri, overwrite=True)

    # initialize tensors
    ds.create_tensor("x")
    ds.create_tensor("images", htype="image", sample_compression="png")

    # add some uncompressed numpy data
    ds.x.append(np.ones((10, 10)))
    
    # add some numpy data and compress as PNG
    data = np.random.randint(low=0, high=256, size=(100, 100, 100, 3), dtype="uint8")
    ds.images.extend(data)


def visualize(uri: str):
    """Visualize some numpy data!"""

    ds = hub.load(uri, read_only=True)

    Image.fromarray(ds.images[0].numpy()).show()
    Image.fromarray(ds.x[0].numpy()).show()


if __name__ == "__main__":
    uri = "./_datasets/npy"
    upload(uri)
    visualize(uri)