import hub

import torch
import pytorch_lightning as pl

from big_net import BigNet


MAX_SAMPLES = 128
TRAIN_URI = "hub://activeloop/mnist-train"
TEST_URI = "hub://activeloop/mnist-test"



def get_train_val_loaders():
    mnist_train = hub.load(TRAIN_URI)[:MAX_SAMPLES]
    mnist_test = hub.load(TEST_URI)[:MAX_SAMPLES]

    def transform(sample):
        x = sample["images"]
        x = x.float().view(-1)
        t = sample["labels"]
        t = t.long()
        return x, t

    train = mnist_train.pytorch(transform=transform, shuffle=False, batch_size=128, num_workers=4)
    val = mnist_test.pytorch(transform=transform, shuffle=False, batch_size=128, num_workers=4)

    return train, val


def train_teacher(model: pl.LightningModule, epochs=1):
    trainer = pl.Trainer(max_epochs=epochs)
    train, val = get_train_val_loaders()
    trainer.fit(model, train, val)


def generate_teacher_embedding_dataset(model: pl.LightningModule, output_dataset_uri: str):
    mnist_train = hub.load(TRAIN_URI)[:MAX_SAMPLES]

    @hub.compute
    def generate_embeddings(sample_in, sample_out):
        x = torch.tensor(sample_in.images.numpy()).view(-1).float()
        y = model(x).detach().numpy()

        sample_out.images.append(y)

        return sample_out

    embeddings = hub.empty(output_dataset_uri, overwrite=True)
    embeddings.create_tensor("images", dtype=float)

    generate_embeddings().eval(mnist_train, embeddings, num_workers=0)


def train_learner():
    # TODO: require `train_teacher` to be called first,
    # TODO: train the learner model on the new local dataset with the teacher embeddings
    raise NotImplementedError


if __name__ == "__main__":
    big_net = BigNet()
    # train_teacher(big_net)

    generate_teacher_embedding_dataset(big_net, "./teacher_embeddings")