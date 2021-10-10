import hub
import numpy as np
from scipy import ndimage

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
