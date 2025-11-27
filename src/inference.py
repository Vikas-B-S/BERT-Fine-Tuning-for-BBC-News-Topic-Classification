import torch
from transformers import BertTokenizer
from src.model import BertLinearProbe
from src.dataset import load_bbc_dataset
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load config
import json
with open("config/params.json") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = BertLinearProbe(model_name=config["model_name"], num_labels=config["num_labels"])
model.load_state_dict(torch.load("results/best_model.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Load BBC dataset to get label names
train_ds, val_ds, test_ds = load_bbc_dataset(model_name=config["model_name"], max_length=config["max_length"])
label_names = test_ds.dataset.classes if hasattr(test_ds, "dataset") else ["business", "entertainment", "politics", "sport", "tech"]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(config["model_name"])

print("Type 'quit' to exit.")

while True:
    text = input("Enter news headline: ").strip()
    if text.lower() == "quit":
        break

    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=config["max_length"])
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).squeeze()

    # Print probabilities
    print("\nClass probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {label_names[i]}: {prob.item():.4f}")

    # Predicted class
    pred_idx = torch.argmax(probs).item()
    print(f"\n✅ Predicted Category: {label_names[pred_idx]}")
    print("--------------------------------")
