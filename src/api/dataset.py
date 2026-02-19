from datasets import load_dataset
from transformers import AutoTokenizer
import os


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
    """
    Датасет для построения RAG-индекса.
    HuggingFace: ruslanmv/ai-medical-chatbot
    Поля: Description (вопрос пациента), Doctor (ответ врача)
    """

    def __init__(self, url="ruslanmv/ai-medical-chatbot", split="train"):
        self.ds = load_dataset(
            path=url, split=split, cache_dir=os.getenv("SAVED_DATASET_QA_PATH")
        )
        print(f'ruslanmv/ai-medical-chatbot" loaded')

    def get_qa_pairs(self) -> list[str]:
        qa_pairs = [f"Q: {ex['Description']} A: {ex['Doctor']}" for ex in self.ds]
        return qa_pairs

    def get_ds(self):
        return self.ds


class RerankerDataset:
    """
    Датасет для обучения BERT-реранкера (cross-encoder).
    HuggingFace: curaihealth/medical_questions_pairs
    Поля: question_1, question_2, label (0/1 — похожи или нет)
    """

    def __init__(self):
        self.ds = load_dataset(
            "curaihealth/medical_questions_pairs",
            cache_dir=os.getenv("SAVED_DATASET_PAIRS_PATH"),
        )
        print(f'Dataset: "curaihealth/medical_questions_pairs" loaded')

    def get_ds(self):
        return self.ds

    def train_test_split(self, test_size: float = 0.2, seed: int = 42):
        return self.ds["train"].train_test_split(test_size=test_size, seed=seed)


def tokenize_reranker(ds, model_name: str = "bert-base-uncased"):
    """Токенизация пар вопросов для cross-encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenization(example):
        return tokenizer(
            example["question_1"],
            example["question_2"],
            truncation=True,
            padding=False,  # DataCollatorWithPadding сделает паддинг в батче
            max_length=512,
        )

    tokenized_ds = ds.map(
        tokenization, batched=True, remove_columns=["question_1", "question_2"]
    )
    return tokenized_ds, tokenizer
