import torch
from torch import nn
from transformers import BertModel

class BertLinearProbe(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # Freeze BERT parameters
        for p in self.bert.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(cls)
