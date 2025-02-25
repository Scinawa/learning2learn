import torch.nn as nn


class CustomArch(nn.Module):
    def __init__(self, non_linearity, nn_architecture=[2, 3, 5, 2, 1]):
        super(CustomArch, self).__init__()

        # Get activation function
        # JI note.
        non_linearity = getattr(nn, non_linearity)

        layers = []
        # Create layers
        for i in range(len(nn_architecture) - 1):
            # Add linear layer
            layers.append(
                nn.Linear(nn_architecture[i], nn_architecture[i + 1], bias=True)
            )

            # Add activation except after last layer
            if i < len(nn_architecture) - 2:
                layers.append(non_linearity())

        # Store layers as ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Apply layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x
