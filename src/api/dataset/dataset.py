from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer


def train_test(ds):
    ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    return ds


def tokenize(ds):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenization(example):
        return tokenizer(
            example["question_1"],
            example["question_2"],
            return_tensors="pt",
            padding=True,
        )

    tokenized_ds = ds.map(tokenization, batched=True)
    return tokenized_ds


class MedicalDataset:
    def __init__(self, url, split):
        self.ds = load_dataset(
            path=url, split=split
        )  # curaihealth/medical_questions_pairs

    def get_qa_pairs(self):
        qa_pairs = [f"Q: {ex['Description']} A: {ex['Doctor']}" for ex in self.ds]
        return qa_pairs

    def get_ds(self):
        return self.ds
