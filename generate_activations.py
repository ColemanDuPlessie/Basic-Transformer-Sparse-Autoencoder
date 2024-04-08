import os

import torch as t
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import Dataset

from transformer import deepspeed_config_location, device
from transformer.model import TinyTransformer
from transformer.config import TinyConfig
from transformer.combined_dataset import get_combined_dataset

from transformer.data_prep import (
    data_collator,
    get_individual_test_dataset,
)


from transformers import Trainer, TrainingArguments

train_dataset, DATA_NUM_STEPS = get_combined_dataset()

MODEL_PATH = "pytorch_model.bin"

test_dataset = get_individual_test_dataset("wikipedia", "20220301.simple") # "wikipedia", "20220301.en"

def gen_mlp_activations(
    model: TinyTransformer,
    train_dataset: TorchIterableDataset,
    test_dataset: Dataset,
):
    tokenizer = model.tokenizer

    checkpoint_dir = "./dataset/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Test the model with predictions

    first_five_inputs = train_dataset[:5]["input_ids"]
    print("Input:", first_five_inputs)
    print("Input:", tokenizer.decode_batch(first_five_inputs))

    for i, input in enumerate(first_five_inputs):
        input: t.Tensor
        activations = model.get_mlp_activations(
            input.unsqueeze(0).to(device)
        )  # (batch_size, seq_len, vocab_size)
    return activations

if __name__ == "__main__":
    model = TinyTransformer()
    model.load_pretrained(MODEL_PATH)
    a = gen_mlp_activations(model, train_dataset, test_dataset)