import hub
import albumentations as A
from albumentations.pytorch import ToTensorV2


augment = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def transform(sample):
    """Sample is an ordered dictionary of dataset elements"""
    image, label = sample["images"], sample["labels"]
    image = augment(image=image)["image"]
    return image, label


def loop():
    # Load the dataset
    ds = hub.load("hub://davitbun/places365-val")

    # Define the dataloader with the transform
    dataloader = ds.pytorch(
        transform=transform,
        num_workers=2,
        batch_size=8,
    )

    # Iterate
    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break


if __name__ == "__main__":
    loop()
