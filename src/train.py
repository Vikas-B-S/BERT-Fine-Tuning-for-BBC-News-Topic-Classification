import torch
from torch.utils.data import DataLoader
from torch import nn
from src.utils import accuracy
from src.dataset import load_bbc_dataset
from src.model import BertLinearProbe
import json
import os

def train(model, train_ds, val_ds, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)

    best_val_loss = float("inf")
    os.makedirs("results/checkpoints", exist_ok=True)

    log_file = open("results/training_log.txt", "w")

    for epoch in range(config["epochs"]):
        print(f"--- Epoch {epoch+1}/{config['epochs']} ---")
        # ---- Training ----
        model.train()
        total_train_loss = 0
        total_train_acc = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += accuracy(logits, labels)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0
        total_val_acc = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_val_loss += loss.item()
                total_val_acc += accuracy(logits, labels)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)

        log_text = (
            f"Epoch {epoch+1}/{config['epochs']}\n"
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}\n"
            f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}\n"
            "---------------------------------------------\n"
        )
        print(log_text)
        log_file.write(log_text)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "results/best_model.pt")

        # Save checkpoint per epoch
        torch.save(model.state_dict(), f"results/checkpoints/epoch_{epoch+1}.pt")

    log_file.close()
    print("Training finished. Best model saved to results/best_model.pt")

if __name__ == "__main__":
    # Load config (correct path even when running as module)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(CURRENT_DIR, "..", "config", "params.json")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)


    # Initialize model
    model = BertLinearProbe(model_name=config["model_name"], num_labels=config["num_labels"])

    # Load datasets
    train_ds, val_ds, test_ds = load_bbc_dataset(model_name=config["model_name"], max_length=config["max_length"])

    # Save test dataset for evaluation
    torch.save(test_ds, "results/test_dataset.pt")

    # Train
    train(model, train_ds, val_ds, config)
