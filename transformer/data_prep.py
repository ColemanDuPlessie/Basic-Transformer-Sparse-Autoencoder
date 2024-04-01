from collections import deque
from collections import deque
from typing import List, Optional

import ray
import torch as t
from datasets import Dataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset as TorchIterableDataset

from transformer import debug
from transformer.config import TinyConfig
from transformer.model import TinyTransformer

model = TinyTransformer()

model_config = TinyConfig()
tokenizer = model.tokenizer


def split_list(input_list: list, num_threads: int) -> list[list[str]]:
    """Split lst into n evenly sized chunks."""
    chunks = []
    thread_size = max(len(input_list) // num_threads, 1)

    for i in range(0, len(input_list), thread_size):
        chunks.append(input_list[i: i + thread_size])
    return chunks


def chunk_examples_ray(
        examples: dict[str, list[str]],
        num_threads: int = 2 if debug else 60,
) -> dict[str, list[list[Optional[int]]]]:
    raw_text_lists = examples["text"]

    split_text_lists = split_list(raw_text_lists, num_threads)
    chunks = ray.get([_chunk_examples.remote(lst) for lst in split_text_lists])

    # Concatenate the chunks
    chunks_list = []
    for chunk in chunks:
        chunks_list += chunk

    return {"input_ids": chunks_list}


def chunk_examples(
        examples: dict[str, list[str]],
) -> dict[str, list[list[Optional[int]]]]:
    raw_text_lists = examples["text"]
    max_length = model_config.max_seq_len

    # Remove Nones from the list
    raw_text_lists = [x for x in raw_text_lists if x is not None]
    tokens = tokenizer.batch_encode_plus(raw_text_lists)["input_ids"]

    chunks: deque[List[Optional[int]]] = deque()
    chunks.append([None] * max_length)
    space_used_in_chunk = 0
    for _article_idx, sentence in enumerate(tokens):  # type: ignore
        while len(sentence) > 0:
            space_available = max_length - space_used_in_chunk

            if len(sentence) < space_available:
                # If the sentence fits in the current chunk, add it
                chunks[-1][
                space_used_in_chunk: space_used_in_chunk + len(sentence) + 1
                ] = sentence + [tokenizer.eos_token_id]
                space_used_in_chunk += len(sentence) + 1
                sentence = []
            else:
                # If the sentence doesn't fit in the current chunk, add as much
                # as possible
                chunks[-1][space_used_in_chunk:] = sentence[:space_available]
                space_used_in_chunk = max_length
                sentence = sentence[space_available:]

            if space_used_in_chunk == max_length:
                # If chunk is full, start a new one
                chunks.append([None] * max_length)
                space_used_in_chunk = 0

    # Pad the last chunk
    chunks[-1][space_used_in_chunk:] = [tokenizer.eos_token_id] * (
            max_length - space_used_in_chunk
    )
    chunks_list = list(chunks)

    return {"input_ids": chunks_list}


@ray.remote
def _chunk_examples(raw_text_lists: list[str]) -> list[list[Optional[int]]]:
    max_length = model_config.max_seq_len
    tokens = tokenizer.batch_encode_plus(raw_text_lists)["input_ids"]

    # print("tokens", tokens[0])
    chunks: deque[List[Optional[int]]] = deque()
    chunks.append([None] * max_length)
    space_used_in_chunk = 0
    for _article_idx, sentence in enumerate(tokens):  # type: ignore
        while len(sentence) > 0:
            space_available = max_length - space_used_in_chunk

            if len(sentence) < space_available:
                # If the sentence fits in the current chunk, add it
                chunks[-1][
                space_used_in_chunk: space_used_in_chunk + len(sentence) + 1
                ] = sentence + [tokenizer.eos_token_id]
                space_used_in_chunk += len(sentence) + 1
                sentence = []
            else:
                # If the sentence doesn't fit in the current chunk, add as much
                # as possible
                chunks[-1][space_used_in_chunk:] = sentence[:space_available]
                space_used_in_chunk = max_length
                sentence = sentence[space_available:]

            if space_used_in_chunk == max_length:
                # If chunk is full, start a new one
                chunks.append([None] * max_length)
                space_used_in_chunk = 0

    # Pad the last chunk
    chunks[-1][space_used_in_chunk:] = [tokenizer.eos_token_id] * (
            max_length - space_used_in_chunk
    )
    chunks_list = list(chunks)

    return chunks_list


def data_collator(batch: list) -> dict:
    # print("batch", batch)

    if tokenizer.eos_token_id is None:
        raise ValueError("Please set EOS token for tokenizer!!!")
    # Extract input_ids from each row in the batch
    input_ids_list = [row["input_ids"] for row in batch]

    try:
        input_ids = t.stack(input_ids_list)
        return {"input_ids": input_ids}
    except:
        # Pad the sequences
        try:
            input_ids = pad_sequence(
                t.tensor(input_ids_list),
                batch_first=True,
                padding_value=tokenizer.eos_token_id,
            )
            return {"input_ids": input_ids}

        except Exception as e:
            print("input_ids_list", input_ids_list)
            for tensor in input_ids_list:
                print(tensor.shape)
            print(e)
            raise RuntimeError("Failed to pad input_ids")


def process_iterable_dataset(raw_dataset: HFIterableDataset) -> TorchIterableDataset:
    print("Processing dataset")
    processed_dataset = raw_dataset.map(
        chunk_examples,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    processed_dataset = processed_dataset.shuffle()
    processed_dataset = processed_dataset.with_format(type="torch")
    assert isinstance(processed_dataset, TorchIterableDataset)  # type: ignore
    train_dataset: TorchIterableDataset = processed_dataset

    return train_dataset


def get_individual_train_dataset(
        data_path: str, data_name: str, data_num_steps: int
) -> TorchIterableDataset:
    # Get streaming train dataset
    dataset = load_dataset(
        path=data_path,
        name=data_name,
        beam_runner="DirectRunner",
        split="train",
        streaming=True,
        # num_proc=32,
    )
    assert isinstance(dataset, HFIterableDataset)

    train_dataset = process_iterable_dataset(dataset)

    return train_dataset


def get_individual_test_dataset(data_path: str, data_name: str) -> Dataset:
    test_dataset_raw: Dataset = load_dataset(
        path=data_path,
        name=data_name,
        beam_runner="DirectRunner",
        split="train[:10]"
        # num_proc=32,
    )  # type: ignore

    processed_test_dataset: Dataset = test_dataset_raw.map(
        chunk_examples,
        batched=True,
        remove_columns=test_dataset_raw.column_names,
        load_from_cache_file=False,
    )  # type: ignore
    processed_test_dataset.set_format(
        type="torch"
        # , columns=["input_ids"]
    )

    return processed_test_dataset
