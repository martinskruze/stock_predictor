import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

class HistoricalStockDataset(Dataset):
    """
    Dataset for historical stock data with price sequences and movement classification labels.
    
    Each sample contains:
    - historical_data: List of 30 days of pricepoint data [pre_market, open, high, low, close, after_hours]
    - price_type: Classification label for price movement (int)
    """
    
    DATASET_DIR = Path(__file__).parent / "source"
    
    def __init__(self, dataset_filename: str):
        """
        Initialize the dataset from a JSON file containing stock data.
        
        Args:
            dataset_filename (str): Filename of the JSON file containing stock data (located in DATASET_DIR)
        """
        self.data = []
        dataset_path = self.DATASET_DIR / dataset_filename
        # Load data from JSON file
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Process each sample
        for sample in raw_data:
            if sample and len(sample) == 2:  # Ensure sample has [historical_data, price_type]
                historical_data, price_type = sample
                
                # Convert historical data to tensor
                # Each day has 6 values: [pre_market, open, high, low, close, after_hours]
                # We'll keep this as a 2D tensor of shape (30, 6) to preserve temporal structure
                if historical_data and len(historical_data) == 30:
                    # Create a 2D tensor of shape (30, 6) - 30 days, 6 values per day
                    data_matrix = []
                    for day_data in historical_data:
                        if len(day_data) == 6:
                            data_matrix.append(day_data)
                    
                    if len(data_matrix) == 30:  # 30 days
                        # Convert to 2D tensor and normalize (optional: you might want to add normalization)
                        data_tensor = torch.tensor(data_matrix, dtype=torch.float32)  # Shape: (30, 6)
                        
                        # Store the data and label
                        self.data.append((data_tensor, price_type))
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.data[idx]


def load_stock_data(num_workers: int = 0, batch_size: int = 32, 
                   shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Load training and validation stock datasets from hardcoded filenames.
    
    Args:
        num_workers (int): Number of worker processes for data loading
        batch_size (int): Batch size for data loading
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders
    """
    train_dataset = HistoricalStockDataset("WDAY_10years_train.json")
    validation_dataset = HistoricalStockDataset("WDAY_10years_validation.json")
    
    train_loader = DataLoader(
        train_dataset, 
        num_workers=num_workers, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=True
    )
    
    validation_loader = DataLoader(
        validation_dataset, 
        num_workers=num_workers, 
        batch_size=batch_size, 
        shuffle=False,  # Usually don't shuffle validation data
        drop_last=True
    )
    
    return train_loader, validation_loader


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy for stock price movement prediction.
    
    Args:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels
        
    Returns:
        torch.Tensor: Accuracy as a single scalar tensor
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return (outputs_idx == labels).float().mean() 