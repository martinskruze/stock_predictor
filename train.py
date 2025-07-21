import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from model_loader import ClassificationLoss, load_model, save_model
from data.historical_stock_dataset import load_stock_data

# copilot (AI) was used for this homework

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 100,
    lr: float = 1e-4,
    batch_size: int = 256,
    seed: int = 2024,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    logger.add_graph(model, torch.zeros(1, 30, 6).to(device)) 
    model.train()

    train_data, val_data = load_stock_data(batch_size=batch_size, num_workers=2, shuffle=True)

    loss_func = ClassificationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )

    global_step = 0
    metrics = {"train_acc": [], "val_acc": [], "train_loss": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for month_stock_data, label in train_data:
            month_stock_data, label = month_stock_data.to(device), label.to(device)

            prediction = model(month_stock_data)
            loss_value = loss_func(prediction, label)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            logger.add_scalar('train_loss', loss_value.item(), global_step)

            accuracy = (prediction.argmax(dim=1) == label).float().mean()
            metrics["train_acc"].append(accuracy)
            metrics["train_loss"].append(loss_value.item())

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for month_stock_data, label in val_data:
                month_stock_data, label = month_stock_data.to(device), label.to(device)
                prediction = model(month_stock_data)
                # compute accuracy
                accuracy = (prediction.argmax(dim=1) == label).float().mean()
                metrics["val_acc"].append(accuracy)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        epoch_train_loss = np.mean(metrics["train_loss"])

        logger.add_scalar('train_accuracy', epoch_train_acc, global_step)
        logger.add_scalar('val_accuracy', epoch_val_acc, global_step)
    
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_loss:.4f} "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True, choices=["linear", "mlp", "mlp_deep", "mlp_deep_residual"])
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--betas", type=lambda s: tuple(map(float, s.split(','))), default=(0.9, 0.999), help="Betas for AdamW optimizer, e.g. '0.9,0.999'")

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
