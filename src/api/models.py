from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import logging

class SimilarityModel:
    def __init__(self, model_name="bert-base-uncased", train=False, device=None,SAVED_MODEL_PATH=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ready = False
        self.SAVED_MODEL_PATH=SAVED_MODEL_PATH

        if train:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            )
            self._ready = True
        else:
            if os.path.exists(self.SAVED_MODEL_PATH):
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.SAVED_MODEL_PATH
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(self.SAVED_MODEL_PATH)
                    self._ready = True
                    logging.info(f"Loaded reranker from {self.SAVED_MODEL_PATH}")
                except Exception as e:
                    logging.warning(
                        f"Failed to load saved_model: {e}. Reranker disabled."
                    )
                    self.model = None
                    self.tokenizer = None
            else:
                logging.warning(
                    f"saved_model not found at '{self.SAVED_MODEL_PATH}'. "
                    "Reranking will use fallback scores. Run training first."
                )
                self.model = None
                self.tokenizer = None

        if self._ready and self.model is not None:
            self.model.to(self.device)

    def predict(self, q1: str, q2: str) -> float:
        # Fallback: return 0.5 if model not loaded
        if self.model is None or self.tokenizer is None:
            return 0.5

        inputs = self.tokenizer(
            q1, q2, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1].item()
