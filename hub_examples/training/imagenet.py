from typing import Tuple
import hub
import torch

from tqdm import tqdm


def transform_sample(sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hub samples are a dictionary that maps a tensor's key -> hub.Tensor."""

    image = sample["images"].numpy()
    label = sample["labels"].numpy()

    x = torch.tensor(image).float()
    x = torch.transpose(x, 0, 2)  # (H, W, C) -> (C, W, H)
    y = torch.tensor(label).int()
    return x, y


if __name__ == "__main__":
    dataloader_workers = 2
    max_samples = 10  # number of samples to use for training
    batch_size = 1
    learning_rate = 0.01
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)

    # TODO: imagenet coming soon!
    imagenet = hub.load("hub://activeloop/imagenet")[:max_samples]
    dataloader = imagenet.pytorch(num_workers=dataloader_workers, transform=transform_sample, batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # feedback: progress bar parameter for pytorch would be nice. saves some boilerplate
    for X, T in dataloader:
        X = X.to(device)
        T = T.to(device)

        optimizer.zero_grad()

        Y = model(X)
        # P = torch.nn.functional.softmax(Y, dim=1)

        # print(X.shape, Y.shape, P.shape, T.shape)

        loss = criterion(Y, T)
        # loss.backward()

        # optimizer.step()
        pass
