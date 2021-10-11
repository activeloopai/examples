import pytorch_lightning as pl

from torch.nn import Sequential, Linear, ReLU, Dropout
from torch.optim import Adam
from torch.nn.functional import cross_entropy


# TODO: paper citation


class BigNet(pl.LightningModule):
    def __init__(self, lr=0.01):
        """Defining the first "big" network.

        PAPER1: 
            "a single large neural net with two hidden layers of 1200 rectified linear hidden units"
            "the net was strongly regularized using dropout and weight-constraints as described in [5]"

        External ref:
            what is a hidden layer? https://medium.com/fintechexplained/what-are-hidden-layers-4f54f7328263

        PAPER2:
            "each hidden unit is randomly omitted from the network with a probability of 0.5"
            NOTE This implementation ignores this step:
                "All layers had L2 weight constraints on the incoming weights of each hidden unit"

        """

        super().__init__()

        self.model = Sequential(
            # NOTE: don't dropout input data (probably a bad idea). exercise for the reader: why?
            Linear(784, 1200),     # hidden layer 1 weights
            ReLU(),
            Dropout(0.5),
            Linear(1200, 1200),    # hidden layer 2 weights
            ReLU(),
            Dropout(0.5),
            Linear(1200, 10),      # output layer weights
        )

        self.critereon = cross_entropy
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.critereon(y_hat, y.view(-1))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.critereon(y_hat, y.view(-1))

        return {"val_loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)