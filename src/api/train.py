import wandb
import torch
from datasets import load_dataset
from src.api.dataset import train_test, tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
        wandb.init(project="medical_bot", name="bert_similarity")

    def train(self):
        split = train_test(self.ds)
        tokenized_datasets = tokenize(split)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        train_dataset = tokenized_datasets["train"].remove_columns(
            [c for c in tokenized_datasets["train"].column_names if c not in columns]
        )
        test_dataset = tokenized_datasets["test"].remove_columns(
            [c for c in tokenized_datasets["test"].column_names if c not in columns]
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir="./saved_model",
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
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model("./saved_model")
        tokenizer.save_pretrained("./saved_model")
