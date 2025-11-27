from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

# ---- Top-level dataset class ----
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.input_ids = torch.stack([x[0] for x in X])
        self.attention_mask = torch.stack([x[1] for x in X])
        self.labels = torch.tensor(y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx]
        }

# ---- Function to load BBC dataset ----
def load_bbc_dataset(model_name, max_length, test_size=0.1, val_size=0.1):
    print("Loading BBC dataset...")
    dataset = load_dataset("SetFit/bbc-news")["train"]

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"],
                         padding="max_length",
                         truncation=True,
                         max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Prepare for sklearn split
    input_ids = [dataset[i]["input_ids"] for i in range(len(dataset))]
    attention_mask = [dataset[i]["attention_mask"] for i in range(len(dataset))]
    labels = [dataset[i]["label"] for i in range(len(dataset))]

    X = list(zip(input_ids, attention_mask))
    y = labels

    # Train / (val + test) split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, random_state=42)
    val_relative = val_size / (test_size+val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_relative, random_state=42)

    # Wrap datasets
    train_ds = CustomDataset(X_train, y_train)
    val_ds = CustomDataset(X_val, y_val)
    test_ds = CustomDataset(X_test, y_test)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    return train_ds, val_ds, test_ds
