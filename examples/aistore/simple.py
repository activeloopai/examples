import hub
import numpy as np

creds = {'endpoint_url': 'http://localhost:51080/s3'}


def create_dataset(path: str):
    """ Create hub dataset and upload random data """

    ds = hub.empty(path, creds=creds, overwrite=True)
    with ds:
        ds.create_tensor("tensor")
        for i in range(10):
            ds.tensor.append(np.random.random((512, 512)))


def loop(path: str):
    """ Load the dataset and stream to pytorch"""
    ds = hub.load(path, creds)
    dataloader = ds.pytorch()

    for (x,) in dataloader:
        print(x)


if __name__ == "__main__":
    path = 's3://ubuntu/dataset'

    create_dataset(path)
    loop(path)
