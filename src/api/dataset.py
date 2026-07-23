from datasets import load_dataset
from transformers import AutoTokenizer
import logging

from src.api.config import get_api_config


def tokenize(ds, model_name: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    def __init__(self, url: str | None = None, split: str | None = None, cache_dir: str | None = None):
        config = get_api_config().dataset
        dataset_name = url or config.qa_name
        dataset_split = split or config.qa_split
        dataset_cache_dir = cache_dir if cache_dir is not None else config.qa_cache_dir
        self.ds = load_dataset(path=dataset_name, split=dataset_split, cache_dir=dataset_cache_dir)
        logging.info(f'Dataset "{dataset_name}" loaded')

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

    def __init__(self, name: str | None = None, cache_dir: str | None = None):
        config = get_api_config().dataset
        dataset_name = name or config.pairs_name
        dataset_cache_dir = cache_dir if cache_dir is not None else config.pairs_cache_dir
        self.ds = load_dataset(dataset_name, cache_dir=dataset_cache_dir)
        logging.info(f'Dataset "{dataset_name}" loaded')

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
            padding=False,
            max_length=512,
        )

    tokenized_ds = ds.map(
        tokenization, batched=True, remove_columns=["question_1", "question_2"]
    )
    return tokenized_ds, tokenizer
