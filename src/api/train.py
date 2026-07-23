from src.api.dataset import RerankerDataset, tokenize_reranker
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.api.config import TrainingConfig, get_api_config


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class ModelTrainer:
    def __init__(self, ds: RerankerDataset, config: TrainingConfig):
        self.reranker_ds = ds
        self.config = config

    def train(self):
        split = self.reranker_ds.train_test_split(test_size=0.2, seed=42)
        tokenized_ds, tokenizer = tokenize_reranker(
            split,
            model_name=self.config.base_model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

        columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        train_dataset = tokenized_ds["train"].remove_columns(
            [c for c in tokenized_ds["train"].column_names if c not in columns]
        )
        eval_dataset = tokenized_ds["test"].remove_columns(
            [c for c in tokenized_ds["test"].column_names if c not in columns]
        )

        print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model_name,
            num_labels=2,
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            report_to=self.config.report_to,
            fp16=self.config.fp16,
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

        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")


if __name__ == "__main__":
    config = get_api_config()
    ds = RerankerDataset(
        name=config.dataset.pairs_name,
        cache_dir=config.dataset.pairs_cache_dir,
    )
    trainer = ModelTrainer(ds, config=config.training)
    trainer.train()
