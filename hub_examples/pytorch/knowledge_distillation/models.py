from torch.nn import Sequential, Linear, ReLU, Dropout


# PAPER1: https://arxiv.org/pdf/1503.02531.pdf (knowledge distillation, section 3)
# PAPER2: https://arxiv.org/pdf/1207.0580.pdf (preventing co-adaption, constraints)


def get_big_net():
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

    return Sequential(
        # NOTE: don't dropout input data (probably a bad idea). exercise for the reader: why?
        Linear(784, 1200),     # hidden layer 1 weights
        ReLU(),
        Dropout(0.5),
        Linear(1200, 1200),    # hidden layer 2 weights
        ReLU(),
        Dropout(0.5),
        Linear(1200, 10),      # output layer weights
    )



def get_small_net():
    # PAPER1: "a smaller net with two hidden layers of 800 rectified linear hidden units and no regularization"
    # NOTE: btw, both of these networks were generated first try by github copilot...... :_)
    
    return Sequential(
        Linear(784, 800),      # hidden layer 1 weights
        ReLU(),
        Linear(800, 800),      # hidden layer 2 weights
        ReLU(),
        Linear(800, 10),       # output layer weights
    )