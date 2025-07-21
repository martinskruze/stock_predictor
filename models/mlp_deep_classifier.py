import torch

class MLPClassifierDeep(torch.nn.Module):
    def __init__(
        self,
        days: int = 30,
        pricepoints: int = 6,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        number_of_channels = days * pricepoints
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(number_of_channels, hidden_dim))
            layers.append(torch.nn.ReLU())
            number_of_channels = hidden_dim
        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, days, pricepoints) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
