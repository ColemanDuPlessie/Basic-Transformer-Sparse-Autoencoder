from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import interleave_datasets, load_dataset
from transformers.trainer_pt_utils import IterableDataset

from transformer.data_prep import process_iterable_dataset


@dataclass
class DatasetStats:
    path: str
    name: Optional[str]
    num_steps: int
    num_rows: int = 0
    proportion_of_our_dataset: float = 0.0
    link: str = ""


datasets = {
    "c4": DatasetStats(
        "c4",
        "en",
        num_steps=20_000,
        num_rows=364_868_892,
        link="https://huggingface.co/datasets/c4",
    ),
    "simple": DatasetStats(
        "wikipedia",
        "20220301.simple",
        num_steps=400,
        num_rows=205_328,
        link="https://huggingface.co/datasets/wikipedia",
    ),
    "english": DatasetStats(
        "wikipedia",
        "20220301.en",
        num_steps=4000,
        num_rows=6_458_670,
        link="https://huggingface.co/datasets/wikipedia",
    ),
    "french": DatasetStats(
        "wikipedia",
        "20220301.fr",
        num_steps=1000,
        link="https://huggingface.co/datasets/wikipedia",
    ),
    "arabic": DatasetStats(
        "graelo/wikipedia",
        "20230901.ar",
        num_steps=1000,
        num_rows=1_210_000,
        link="https://huggingface.co/datasets/graelo/wikipedia",
    ),
    "yoruba": DatasetStats(
        "graelo/wikipedia",
        "20230601.yo",
        num_steps=300,
        num_rows=32_000,
        link="https://huggingface.co/datasets/graelo/wikipedia/viewer/20230601.yo",
    ),
    "japanese": DatasetStats(
        "graelo/wikipedia",
        "20230601.ja",
        num_steps=700,
        num_rows=990_000,
        link="https://huggingface.co/datasets/graelo/wikipedia/viewer/20230601.ja",
    ),
    "tiny_stories": DatasetStats(
        "roneneldan/TinyStories",
        None,
        num_steps=1000,
        num_rows=2_120_000,
        link="https://huggingface.co/datasets/roneneldan/TinyStories",
    ),
}

total_num_steps = sum([dataset.num_steps for dataset in datasets.values()])
total_num_rows = sum([dataset.num_rows for dataset in datasets.values()])


def get_combined_dataset() -> Tuple[IterableDataset, int]:
    raw_datasets = []
    dataset_probs = []
    for dataset in datasets.values():
        raw_datasets.append(
            load_dataset(
                path=dataset.path,
                name=dataset.name,
                # split=f"train[0:{dataset.num_rows}]",
                split="train",
                streaming=True,
            )
        )
        #
        if dataset.proportion_of_our_dataset:
            dataset_probs.append(dataset.proportion_of_our_dataset)
        else:
            dataset_probs.append(dataset.num_steps / total_num_steps)
        print(f"{dataset.path} - {dataset.name}: {dataset_probs[-1] * 100:.2f}%")

    full_raw_dataset = interleave_datasets(raw_datasets, probabilities=dataset_probs)

    train_dataset = process_iterable_dataset(full_raw_dataset)

    return train_dataset, total_num_steps


def get_en_fr_dataset() -> Tuple[IterableDataset, int]:
    en_wiki = load_dataset(
        path="wikipedia",
        name="20220301.en",
        split="train",
        streaming=True,
    )
    fr_wiki = load_dataset(
        path="wikipedia",
        name="20220301.fr",
        split="train",
        streaming=True,
    )

    full_raw_dataset = interleave_datasets([en_wiki, fr_wiki], probabilities=[0.5, 0.5])  # type: ignore
    train_dataset = process_iterable_dataset(full_raw_dataset)
    return train_dataset, 100_000


def get_en_code_dataset() -> Tuple[IterableDataset, int]:
    code = load_dataset(
        path="codeparrot/github-code",
        streaming=True,
        split="train",
        languages=["Python"],
        licenses=["mit", "isc"],
    )
    # Rename code column to text
    code = code.rename_column("code", "text")

    en_wiki = load_dataset(
        path="wikipedia",
        name="20220301.en",
        split="train",
        streaming=True,
    )

    full_raw_dataset = interleave_datasets([code, en_wiki], probabilities=[0.5, 0.5])  # type: ignore
    train_dataset = process_iterable_dataset(full_raw_dataset)
    return train_dataset, 100_000
