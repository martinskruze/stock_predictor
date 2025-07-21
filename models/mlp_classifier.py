import torch

class MLPClassifier(torch.nn.Module):
    def __init__(
        self,
        days: int = 30,
        pricepoints: int = 6,
        num_classes: int = 7,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(days * pricepoints, 256))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(256, num_classes))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, days, pricepoints) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)