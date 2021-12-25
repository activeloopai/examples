import hub  # type: ignore
from hub.core.dataset import Dataset  # type: ignore
from PIL import Image, ImageFilter, ImageOps, ImageEnhance  # type: ignore
import random
from typing import Tuple


@hub.compute
def cvt_horizontal_flip(
    sample_input: Dataset, sample_output: Dataset, probability: float = 0.5
) -> Dataset:
    """Converts the sample_input dataset to horizontal flips in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_vertical_flip(
    sample_input: Dataset, sample_output: Dataset, probability: float = 0.5
) -> Dataset:
    """Converts the sample_input dataset to vertical flips in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_blur(
    sample_input: Dataset,
    sample_output: Dataset,
    blur_value: float = 0,
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to blurs in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        blur_value: value to determine the extent of blur. Defaults to 0.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.filter(ImageFilter.BoxBlur(blur_value))
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_gray(
    sample_input: Dataset, sample_output: Dataset, probability: float = 0.5
) -> Dataset:
    """Converts the sample_input dataset to grayscale in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = ImageOps.grayscale(img)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_inverse(
    sample_input: Dataset, sample_output: Dataset, probability: float = 0.5
) -> Dataset:
    """Converts the sample_input dataset to inverts in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = ImageOps.invert(img)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_contrast(
    sample_input: Dataset,
    sample_output: Dataset,
    contrast_value: float = 1,
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to contrasts in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        contrast_value: value to determine the extent of contrast. Defaults to 1.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = ImageEnhance.Contrast(img).enhance(contrast_value)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_crop(
    sample_input: Dataset,
    sample_output: Dataset,
    crop_locations: Tuple[int, int, int, int] = None,
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to crops in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        crop_locations: tuple (start_x,start_y,end_x,end_y) to determine region for crop. Defaults to -1 which causes a centre crop.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if crop_locations is None:
        crop_locations = (
            img.size[0] * 0.25,
            img.size[1] * 0.25,
            img.size[0] * 0.75,
            img.size[1] * 0.75,
        )
    if random.uniform(0, 1) < probability:
        img = img.crop(crop_locations)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_resize(
    sample_input: Dataset,
    sample_output: Dataset,
    resize_size: Tuple[int, int] = None,
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to resizes in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        resize_size: tuple (width,height) to determine dimensions for resize. Defaults to -1 which prevents resizing.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if resize_size is None:
        resize_size = (img.size[0], img.size[1])
    if random.uniform(0, 1) < probability:
        img = img.resize(resize_size)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_rotate(
    sample_input: Dataset,
    sample_output: Dataset,
    rotate_angle: float = 0,
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to rotations in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        rotate_angle: value to determine extent of angular rotation. Defaults to 0 which prevents rotation.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.rotate(rotate_angle)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_transpose(
    sample_input: Dataset,
    sample_output: Dataset,
    transpose_value: int = 0,
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to transpose in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        transpose_value: value to determine type of transpose.
            {
                0: Transpose top,
                90: Transpose right,
                180: Transpose bottom,
                270: Transpose left
            }
            Defaults to 0 which prevents transpose.
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    values = {0: 0, 90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability and transpose_value in values:
        img = img.transpose(values[transpose_value])
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_padding(
    sample_input: Dataset,
    sample_output: Dataset,
    pad_size: Tuple[int, int, int, int] = (0, 0, 0, 0),
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    probability: float = 0.5,
) -> Dataset:
    """Converts the sample_input dataset to padded in sample_output dataset.

    Args:
        sample_input: input dataset passed to generate output dataset.
        sample_output: output dataset which will contain transforms of input dataset.
        pad_size: tuple (pad_left,pad_top,pad_right,pad_bottom) to determine dimensions of padding. Defaults to (0,0,0,0) which prevents padding.
        pad_color: tuple (r,g,b) in rgb format to set the color of padding. Defaults to black (0,0,0).
        probability: probability to randomly apply transformation. Defaults to 0.5.

    Returns:
        sample_output dataset with transformed images.
    """
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        new_img = Image.new(
            img.mode,
            (
                pad_size[0] + img.size[0] + pad_size[2],
                pad_size[1] + img.size[1] + pad_size[3],
            ),
            pad_color,
        )
        new_img.paste(img, (pad_size[0], pad_size[1]))
    sample_output.images.append(new_img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


if __name__ == "__main__":

    ds_input = hub.load("/input/path")
    ds_output = hub.like("/output/path", ds_input)
    pipeline = hub.compose(
        [
            cvt_horizontal_flip(probability=0.4),
            cvt_crop(crop_locations=-1, probability=0.8),
            cvt_gray(probability=0.7),
            cvt_padding(pad_size=(10, 10, 10, 10), bg_color=(0, 0, 0), probability=0.5),
            cvt_resize(resize_size=(100, 80), probability=0.6),
        ]
    )
    pipeline.eval(ds_input, ds_output, num_workers=2)
