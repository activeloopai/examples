import torch

# NOTE: `PAPER:` denotes a quote from the cited paper.

# reference paper: https://arxiv.org/pdf/1503.02531.pdf
# this script implements section 3: "Preliminary experiments on MNIST"

# PAPER: "a single large neural net with two hidden layers of 1200 rectified linear hidden units"
# what is a hidden layer? https://medium.com/fintechexplained/what-are-hidden-layers-4f54f7328263
net = torch.nn.Sequential(
    torch.nn.Linear(784, 1200),     # hidden layer 1 weights
    torch.nn.ReLU(),
    torch.nn.Linear(1200, 1200),    # hidden layer 2 weights
    torch.nn.ReLU(),
    torch.nn.Linear(1200, 10),      # output layer weights
)

big_net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

# PAPER: "a smaller net with two hidden layers of 800 rectified linear hidden units and no regularization"
# TODO

# PAPER: "the net was strongly regularized using dropout and weight-constraints as described in [5]"
# TODO


# PAPER: "in addition, the input images were jittered by up to two pixels in any direction"
# TODO
