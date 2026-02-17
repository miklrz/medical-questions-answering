import wandb
import torch
from datasets import load_dataset
from src.api.dataset import RerankerDataset, tokenize_reranker
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logger
import os

BASE_MODEL = "bert-base-uncased"
SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_PATH", "./saved_model")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class ModelTrainer:
    def __init__(self, ds):
        self.ds = ds.get_ds()
        wandb.init(project="medical_bot", name="bert_reranker_cross_encoder")
        self.reranker_ds = RerankerDataset()

    def train(self):
        split = self.reranker_ds.train_test_split(test_size=0.2, seed=42)
        tokenized_ds, tokenizer = tokenize_reranker(split, model_name=BASE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        train_dataset = tokenized_ds["train"].remove_columns(
            [c for c in tokenized_ds["train"].column_names if c not in columns]
        )
        eval_dataset = tokenized_ds["test"].remove_columns(
            [c for c in tokenized_ds["test"].column_names if c not in columns]
        )

        logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=SAVED_MODEL_PATH,
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=20,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="wandb",
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("Training BERT cross-encoder reranker...")
        trainer.train()

        trainer.save_model(SAVED_MODEL_PATH)
        tokenizer.save_pretrained(SAVED_MODEL_PATH)
        print(f"Model saved to {SAVED_MODEL_PATH}")

        wandb.finish()
