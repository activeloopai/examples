import hub
import numpy as np
from PIL import Image
import argparse
import tqdm
import time
import torchvision.datasets as datasets

NUM_WORKERS = 1
DS_OUT_PATH = "./data/places365"  # optionally s3://, gcs:// or hub:// path
DOWNLOAD = False

parser = argparse.ArgumentParser(description="Hub Places365 Uploading")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--num_workers",
    type=int,
    default=NUM_WORKERS,
    metavar="O",
    help="number of workers to allocate",
)
parser.add_argument(
    "--ds_out",
    type=str,
    default=DS_OUT_PATH,
    metavar="O",
    help="dataset path to be transformed into",
)

parser.add_argument(
    "--download",
    type=bool,
    default=DOWNLOAD,
    metavar="O",
    help="Download from the source http://places2.csail.mit.edu/download.html",
)

args = parser.parse_args()


def define_dataset(path: str, class_names: list = []):
    ds = hub.empty(path, overwrite=True)

    ds.create_tensor("images", htype="image", sample_compression="jpg")
    ds.create_tensor("labels", htype="class_label", class_names=class_names)

    return ds


@hub.compute
def upload_parallel(pair_in, sample_out):
    filepath, target = pair_in[0], pair_in[1]
    img = Image.open(filepath)
    if len(img.size) == 2:
        img = img.convert("RGB")
    arr = np.asarray(img)
    sample_out.images.append(arr)
    sample_out.labels.append(target)


def upload_iteration(filenames_target: list, ds: hub.Dataset):
    with ds:
        for filepath, target in tqdm.tqdm(filenames_target):
            img = Image.open(filepath)
            if len(img.size) == 2:
                img = img.convert("RGB")
            arr = np.asarray(img)
            ds.images.append(arr)
            ds.labels.append(target)


if __name__ == "__main__":

    for split in [
        "val",
        "train-standard",
    ]:  # optionally add ["train-challenge"] for 8M images
        torch_dataset = datasets.Places365(
            args.data,
            split=split,
            download=args.download,
        )
        categories = torch_dataset.load_categories()[0]
        categories = list(map(lambda x: "/".join(x.split("/")[2:]), categories))
        ds = define_dataset(f"{args.ds_out}-{split}", categories)
        filenames_target = torch_dataset.load_file_list()

        print(f"uploading {split}...")
        t1 = time.time()
        if args.num_workers > 1:

            upload_parallel().eval(
                filenames_target[0],
                ds,
                num_workers=args.num_workers,
                scheduler="processed",
            )
        else:
            upload_iteration(filenames_target[0], ds)
        t2 = time.time()
        print(f"uploading {split} took {t2-t1}s")
