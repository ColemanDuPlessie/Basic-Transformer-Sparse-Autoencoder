import os

import matplotlib.pyplot as plt

import torch as t
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import Dataset

from autoencoder import TinyAutoencoder

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
            
LEARNING_RATE = 0.001
REGULARIZATION_VALUE = 0.0001
PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 4096
MAX_SAMPLES = 262144
BATCH_SIZE = 256

def train_one_epoch(autoencoder, optimizer, data):
    optimizer.zero_grad()
    loss, out = autoencoder(data)
    loss.backward()
    optimizer.step()
    return loss, out
            
def main():

    pretrained_model = TinyTransformer().to(device)
    pretrained_model.load_pretrained(MODEL_PATH)
    pretrained_model.eval()
    
    autoencoder = TinyAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE).to(device)
    optimizer = t.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=0.001) # TODO tune these a lot
    
    activations = []
    losses = []
    for i, input in enumerate(train_dataset):
    	if i == MAX_SAMPLES: break
    	in_tensor = input["input_ids"]
    	activations.append(pretrained_model.get_mlp_activations(
    	in_tensor.unsqueeze(0).to(device)).squeeze()) # A tensor of shape [input_len, hidden_dim]
    	if len(activations) == BATCH_SIZE:
    	    loss, out = train_one_epoch(autoencoder, optimizer, t.stack(activations, dim=0))
    	    activations = []
    	    losses.append(loss.detach().item())
    	    print(f"Completed batch {i//BATCH_SIZE} with loss {loss.detach().item()}.")
    plt.plot(losses)
    plt.show()
    t.save(autoencoder, "trained_autoencoder.pt")

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
