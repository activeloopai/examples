import pytorch_lightning as pl

from torch.optim import Adam
from torch.nn.functional import mse_loss, cross_entropy


# PAPER1: https://arxiv.org/pdf/1503.02531.pdf (knowledge distillation, section 3)
# PAPER2: https://arxiv.org/pdf/1207.0580.pdf (preventing co-adaption, constraints)


class Learner(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        """Knowledge distillation learner. The incoming model will
        be trained on the image embeddings from the trained `Teacher` model.
        """

        super().__init__()

        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = mse_loss(y_hat, y)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = cross_entropy(y_hat, y.view(-1))

        return {"val_loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)