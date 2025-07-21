import torch

class LinearClassifier(torch.nn.Module):
    def __init__(
        self,
        days: int = 30,
        pricepoints: int = 6,
        num_classes: int = 7,
    ):
        """
        Args:
            days: int, how many days of data to use
            pricepoints: int, how many price points to use
            num_classes: int, number of classes
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(days * pricepoints, num_classes))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, days, pricepoints)

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
