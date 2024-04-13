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

SAVED_DATA_PATH = "./dataset/MLP_activations_"
SAVED_DATA_EXTENSION = ".pt"
SAVE_BATCH_SIZE = 256

test_dataset = get_individual_test_dataset("wikipedia", "20220301.simple") # "wikipedia", "20220301.en"

def gen_mlp_activations(
    model: TinyTransformer,
    dataset: TorchIterableDataset,
    save_path, save_extension, save_batch_size,
    num_samples = -1
):
    tokenizer = model.tokenizer

    checkpoint_dir = "./dataset/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    out = []
    
    for i, input in enumerate(dataset):
        if i == num_samples: break
        input_tensor = input["input_ids"]
        activations = model.get_mlp_activations(
            input_tensor.unsqueeze(0).to(device)
        )  # (batch_size, seq_len, vocab_size)
        out.append(activations.detach())
        if i % save_batch_size == save_batch_size-1:
            saving = t.stack(out, dim=0)
            t.save(saving, save_path + str(i-save_batch_size+1) + "-" + str(i) + save_extension)
            out = []
            saving = None
            print("Saving a batch starting with element number " + str(i))

if __name__ == "__main__":
    model = TinyTransformer()
    model.load_pretrained(MODEL_PATH)
    model = model.to(device)
    gen_mlp_activations(model, train_dataset, SAVED_DATA_PATH, SAVED_DATA_EXTENSION, SAVE_BATCH_SIZE)
    print("Finished successfully!")
