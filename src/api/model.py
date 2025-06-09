from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SimilarityModel:
    def __init__(self, model_name="bert-base-uncased", train=False, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if train:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "./saved_model"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("./saved_model")
        self.model.to(self.device)

    def predict(self, q1: str, q2: str) -> float:
        inputs = self.tokenizer(
            q1, q2, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1].item()
