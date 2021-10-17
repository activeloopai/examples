import argparse
import hub
import numpy as np
from scipy import ndimage


parser = argparse.ArgumentParser(
    description="Image Transformation/Augmentation Using hub.compute Decorator"
)

parser.add_argument(
    "--ds",
    type=str,
    default="hub://activeloop/mnist-train",  # optionally s3://, gcs:// or hub:// path
    metavar="O",
    help="dataset path which will be transformed",
)

parser.add_argument(
    "--ds_out",
    type=str,
    default="./transformed_dataset",
    metavar="O",
    help="dataset path to be transformed into",
)

args = parser.parse_args()

@hub.compute
def random_vflip(sample_in, sample_out, p=0.5):
    img = sample_in.images.numpy()
    if np.random.randn(1) < p:
      img = np.flip(img, axis=1)

    sample_out.labels.append(sample_in.labels.numpy())
    sample_out.images.append(img)
    return sample_out


@hub.compute
def random_hflip(sample_in, sample_out, p=0.5):
    img = sample_in.images.numpy()
    if np.random.randn(1) < p:
      img = np.flip(img, axis=0)

    sample_out.labels.append(sample_in.labels.numpy())
    sample_out.images.append(img)
    return sample_out


@hub.compute
def rotate(sample_in, sample_out, angle=45):
    sample_out.labels.append(sample_in.labels.numpy())
    sample_out.images.append(ndimage.rotate(sample_in.images.numpy(), angle))

    return sample_out


@hub.compute
def resize(sample_in, sample_out, new_size):
    sample_out.labels.append(sample_in.labels.numpy())
    sample_out.images.append(np.array(Image.fromarray(
        sample_in.images.numpy()).resize(new_size)))

    return sample_out


@hub.compute
def crop_center(sample_in, sample_out, crop_h=0.5, crop_w=0.5):

    img = sample_in.images.numpy()

    h, w = img.shape[0], img.shape[1]

    crop_h, crop_w = int(h * crop_h), int(w * crop_w)
    start_h = h//2-(crop_h//2)
    start_w = w//2-(crop_w//2)

    if img.ndim == 2:
       sample_out.images.append(img[start_h:h-start_h, start_w:h-start_w])
    if img.ndim == 3:
       sample_out.images.append(img[start_h:h-start_h, start_w:h-start_w, :])
    sample_out.labels.append(sample_in.labels.numpy())
    return sample_out



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

