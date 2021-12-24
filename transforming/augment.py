import hub
import random
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


@hub.compute
def cvt_horizontal_flip(sample_input, sample_output, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_vertical_flip(sample_input, sample_output, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_blur(sample_input, sample_output, blur_value=0, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.filter(ImageFilter.BoxBlur(blur_value))
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_gray(sample_input, sample_output, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = ImageOps.grayscale(img)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_inverse(sample_input, sample_output, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = ImageOps.invert(img)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_contrast(sample_input, sample_output, contrast_value=1, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = ImageEnhance.Contrast(img).enhance(contrast_value)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_crop(sample_input, sample_output, crop_locations=-1, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if crop_locations == -1:
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
def cvt_resize(sample_input, sample_output, resize_size=-1, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if resize_size == -1:
        resize_size = (img.size[0], img.size[1])
    if random.uniform(0, 1) < probability:
        img = img.resize(resize_size)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_rotate(sample_input, sample_output, rotate_angle=0, probability=0.5):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        img = img.rotate(rotate_angle)
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_transpose(sample_input, sample_output, transpose_value=0, probability=0.5):

    values = {0: 0, 90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability and transpose_value in values:
        img = img.transpose(values[transpose_value])
    sample_output.images.append(img)
    sample_output.labels.append(sample_input.labels.numpy())
    return sample_output


@hub.compute
def cvt_padding(
    sample_input,
    sample_output,
    pad_size=(0, 0, 0, 0),
    bg_color=(0, 0, 0),
    probability=0.5,
):

    img = Image.fromarray(sample_input.images.numpy())
    if random.uniform(0, 1) < probability:
        new_img = Image.new(
            img.mode,
            (
                pad_size[0] + img.size[0] + pad_size[2],
                pad_size[1] + img.size[1] + pad_size[3],
            ),
            bg_color,
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