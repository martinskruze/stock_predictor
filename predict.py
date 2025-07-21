import argparse
from model_loader import load_model
from utilites.price_classifier import PriceClassifier
import torch
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["linear", "mlp", "mlp_deep", "mlp_deep_residual"])
    args = parser.parse_args()

    # Load model with weights
    model = load_model(args.model_name, with_weights=True)
    model.eval()

    # Use CPU for inference
    device = torch.device("cpu")
    model.to(device)

    # Load the real future datapoint (30x6 matrix)
    datapath = Path(__file__).parent / "data" / "source" / "WDAY_real_future.json"
    with open(datapath, "r") as f:
        data_matrix = json.load(f)

    data_tensor = torch.tensor(data_matrix, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, 30, 6)

    with torch.no_grad():
        logits = model(data_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_class = logits.argmax(dim=1).item()
        description = PriceClassifier.description(pred_class)

        print("Class probabilities:")
        for i, prob in enumerate(probabilities):
            desc = PriceClassifier.description(i)
            print(f"  Class {i}: {prob*100:.2f}% - {desc}")
        print(f"\nPredicted class = {pred_class}, Description = {description}") 