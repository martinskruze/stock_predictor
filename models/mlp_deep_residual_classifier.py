import torch

class MLPClassifierDeepResidual(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features, bias=False),
                torch.nn.LayerNorm(out_features),
                torch.nn.ReLU()
            )
            if in_features != out_features:
                self.skip = torch.nn.Linear(in_features, out_features, bias=False)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)

    def __init__(
        self,
        days: int = 30,
        pricepoints: int = 6,
        num_classes: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 8,
    ):
        """
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

        layers.append(torch.nn.Linear(days * pricepoints, hidden_dim))
        for _ in range(num_layers):
            layers.append(self.Block(hidden_dim, hidden_dim))
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
