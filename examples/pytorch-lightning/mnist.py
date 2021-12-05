"""pytorch-lightning example with Hub dataloaders
Based on this colab:
https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31
"""

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
import hub


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # called with self(x)
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return hub.load("hub://activeloop/mnist-train").pytorch(
            batch_size=32,
            num_workers=2,
            use_local_cache=True,
            transform=tranform,
        )

    def val_dataloader(self):
        # OPTIONAL
        return hub.load("hub://activeloop/mnist-test").pytorch(
            batch_size=32,
            num_workers=0,
            use_local_cache=True,
            transform=tranform,
        )

    def test_dataloader(self):
        # OPTIONAL
        return hub.load("hub://activeloop/mnist-test").pytorch(
            batch_size=32,
            num_workers=0,
            use_local_cache=True,
            transform=tranform,
        )

# outside the class to make it pickalable
# formats the data to meet the input layer 
def tranform(x):
    return x["images"][None, :, :].astype("float32"), x["labels"][0]


if __name__ == "__main__":

    mnist_model = MNISTModel()
    trainer = pl.Trainer(gpus=0, max_epochs=2, strategy="ddp")
    trainer.fit(mnist_model)
    trainer.test(mnist_model)
