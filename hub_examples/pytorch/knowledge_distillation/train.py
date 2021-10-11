import hub

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from teacher import Teacher
from learner import Learner
from models import get_big_net, get_small_net


# PAPER1: https://arxiv.org/pdf/1503.02531.pdf (knowledge distillation, section 3)
# PAPER2: https://arxiv.org/pdf/1207.0580.pdf (preventing co-adaption, constraints)


MAX_SAMPLES = 60_000
TRAIN_URI = "hub://activeloop/mnist-train"
TEST_URI = "hub://activeloop/mnist-test"
EMBEDDINGS_URI = "./_datasets/teacher_embeddings"



def get_teacher_loaders():
    mnist_train = hub.load(TRAIN_URI)[:MAX_SAMPLES]
    mnist_test = hub.load(TEST_URI)[:MAX_SAMPLES]

    def transform(sample):
        x = sample["images"]
        x = x.float().view(-1)
        t = sample["labels"]
        t = t.long()
        return x, t

    train = mnist_train.pytorch(transform=transform, shuffle=True, batch_size=128, num_workers=4)
    val = mnist_test.pytorch(transform=transform, shuffle=False, batch_size=128, num_workers=4)

    return train, val


def get_learner_loaders():
    mnist_embeddings = hub.load(EMBEDDINGS_URI)[:MAX_SAMPLES]

    def transform(sample):
        x = sample["images"]
        x = x.float().view(-1)
        t = sample["labels"]
        t = t.float()
        return x, t

    train = mnist_embeddings.pytorch(transform=transform, shuffle=True, batch_size=128, num_workers=4)

    _, val = get_teacher_loaders()

    return train, val

def train_teacher(model: pl.LightningModule, epochs=1):
    trainer = pl.Trainer(max_epochs=epochs, logger=WandbLogger())
    train, val = get_teacher_loaders()
    trainer.fit(model, train, val)


def generate_teacher_embedding_dataset(model):
    mnist_train = hub.load(TRAIN_URI)[:MAX_SAMPLES]

    @hub.compute
    def generate_embeddings(sample_in, sample_out):
        image = sample_in.images.numpy()

        x = torch.tensor(image).view(-1).float()
        y = model(x).detach().numpy()

        sample_out.images.append(image)
        sample_out.labels.append(y)

        return sample_out

    embeddings = hub.empty(EMBEDDINGS_URI, overwrite=True)
    embeddings.create_tensor("images", dtype="uint8")
    embeddings.create_tensor("labels", dtype=float)

    generate_embeddings().eval(mnist_train, embeddings, num_workers=0)


def train_learner(model: pl.LightningModule, epochs=1):
    trainer = pl.Trainer(max_epochs=epochs, logger=WandbLogger())
    train, val = get_learner_loaders()
    trainer.fit(model, train, val)


if __name__ == "__main__":
    # first, we need to train the teacher network
    print("\n\nTraining teacher\n\n")
    big_net = get_big_net()
    train_teacher(Teacher(big_net))

    # now the teacher network is trained. let's generate a new hub dataset.
    # this new dataset doesn't change the `images` tensor (it just copies it)
    # but it DOES change the `labels` tensor. instead of the normal mnist labels,
    # it uses the embeddings (outputs for each `images` sample) of the teacher model
    # print("\n\nGenerating embedding dataset\n\n")
    # generate_teacher_embedding_dataset(big_net)

    # finally, we can train the learner network to predict the output embeddings
    # of the teacher network. we can do so by using the new output embeddings dataset
    # print("\n\nTraining learner on embedding dataset\n\n")
    # small_net = get_small_net()
    # train_learner(Learner(small_net))