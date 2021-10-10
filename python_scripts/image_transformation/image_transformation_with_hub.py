import argparse
import hub
from transformations import random_vflip, rotate

dataset_path = "hub://activeloop/mnist-train"
transformed_dataset_path = (
    "./transformed_dataset"  # optionally s3://, gcs:// or hub:// path
)

parser = argparse.ArgumentParser(
    description="Image Transformation/Augmentation Using hub.compute Decorator"
)

parser.add_argument(
    "--ds",
    type=str,
    default=dataset_path,
    metavar="O",
    help="dataset path which will be transformed",
)

parser.add_argument(
    "--ds_out",
    type=str,
    default=transformed_dataset_path,
    metavar="O",
    help="dataset path to be transformed into",
)

args = parser.parse_args()


if __name__ == "__main__":

    # loading dataset
    ds = hub.load(args.ds)

    # hub.like is used to create an empty dataset with the same tensor structure.
    ds_transformed = hub.like(args.ds_out, ds)

    # creating pipeline for modularizing dataset processing.
    pipeline = hub.compose([random_vflip(p=0.7), rotate(angle=20)])

    # evaluating whole pipeline for first 5 samples of dataset. You can also give whole dataset for evaluating.
    pipeline.eval(ds[:6], ds_transformed, num_workers=2)

    print("The transformed dataset has been created successfully.")