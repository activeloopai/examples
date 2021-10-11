import hub
import pytorch_lightning as pl

from torch.optim import Adam
from torch.nn.functional import cross_entropy


# PAPER1: https://arxiv.org/pdf/1503.02531.pdf (knowledge distillation, section 3)
# PAPER2: https://arxiv.org/pdf/1207.0580.pdf (preventing co-adaption, constraints)


class Teacher(pl.LightningModule):
    def __init__(self, model, lr=0.01):
        """Knowledge distillation teacher. The incoming model will
        be trained on the actual images. After it is trained, the
        learner model will be trained on the final embeddings of the incoming
        model to this class.
        """

        super().__init__()

        self.model = model
        self.loss = cross_entropy

        self.hparams.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view(-1))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view(-1))

        return {"val_loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)