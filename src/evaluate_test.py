# evaluate_test.py

import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from src.model import BertLinearProbe
from src.dataset import CustomDataset  # Your custom dataset
import json
import warnings

# ----------------------------
# Load config
# ----------------------------
with open("config/params.json") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------
# Load model safely
# ----------------------------
model = BertLinearProbe(model_name=config["model_name"], num_labels=config["num_labels"])

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    model.load_state_dict(torch.load("results/best_model.pt", map_location=device))

model.to(device)
model.eval()

# ----------------------------
# Load test dataset safely
# ----------------------------
test_dataset_path = "results/test_dataset.pt"

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with torch.serialization.safe_globals([CustomDataset]):
        test_ds = torch.load(test_dataset_path, weights_only=False)

test_loader = DataLoader(test_ds, batch_size=config["batch_size"], num_workers=0)

# ----------------------------
# Loss function
# ----------------------------
criterion = nn.CrossEntropyLoss()

# ----------------------------
# Evaluate
# ----------------------------
all_preds = []
all_labels = []
total_loss = 0

print("\nEvaluating on test set...\n")
for batch in tqdm(test_loader, desc="Batches", unit="batch"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# ----------------------------
# Compute Metrics
# ----------------------------
avg_loss = total_loss / len(test_loader)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

# ----------------------------
# Print Results
# ----------------------------
print("\n===== Test Set Evaluation =====")
print(f"Loss      : {avg_loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("================================\n")
