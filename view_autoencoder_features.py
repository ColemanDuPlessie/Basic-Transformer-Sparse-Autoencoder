import os

from operator import itemgetter

import torch as t
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import Dataset, load_dataset

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

HELP_MESSAGE = "To see the n tokens (as well as their indices) that most activate feature #f, call 'FEATURE f n'.\nTo see the full text of string n, call 'STRING n'.\nTo see only selected tokens from string n, call 'STRING n start stop'.\nTo close the program, call 'EXIT'.\nYou may input 'HELP' at any time to see this message again\n"

MODEL_PATH = "pytorch_model.bin"
AUTOENCODER_PATH = "trained_autoencoder.pt"
PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 4096

test_dataset = load_dataset(path="roneneldan/TinyStories", name=None, split="validation", streaming=True) # "wikipedia", "20220301.en"
            
TOTAL_SAMPLES = 256

CONTEXT_VIEW_SIZE = 10
HIGHLIGHT_COLOR = '\033[92m'
HIGHLIGHT_END = '\033[0m'

def find_str_idx(str_idxs, target_idx): # TODO this is almost certainly not the best way to do this
    target_str = len(str_idxs)//2
    if str_idxs[target_str] <= target_idx and (target_str+1 == len(str_idxs) or str_idxs[target_str+1] > target_idx):
        return target_str, target_idx-str_idxs[target_str]
    elif str_idxs[target_str] > target_idx:
        return find_str_idx(str_idxs[:target_str], target_idx)
    else:
        ans = find_str_idx(str_idxs[target_str+1:], target_idx)
        return ans[0]+target_str+1, ans[1]

def get_max_acts(acts, feat, qty=1):
    """
    Returns a namedtuple (as from torch.topk) of values and indices. Acts should be a 2D tensor of shape [num_tokens, num_features].
    """
    feat_tensor = t.select(acts, 1, feat)
    return t.topk(feat_tensor, qty, dim=0)
            
def main():

    print("Loading pretrained models...")
    
    pretrained_model = TinyTransformer().to(device)
    pretrained_model.load_pretrained(MODEL_PATH)
    pretrained_model.eval()
    
    # autoencoder = TinyAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE).to(device)
    # autoencoder.load_pretrained(AUTOENCODER_PATH)
    # autoencoder.eval() TODO this is the right way, but I saved the autoencoder wrong...
    
    autoencoder = t.load(AUTOENCODER_PATH, map_location=device)
    
    print(f"Models loaded successfully! Now finding features for {TOTAL_SAMPLES} strings from the test set...")
    
    activations = [] # List of tensors of shape [input_len, autoencoder_hidden_dim]
    text = [] # List of lists of tokens, each corresponding to an activation
    text_starts = [0]
    for i, in_data in enumerate(test_dataset):
    	if i == TOTAL_SAMPLES: break
    	in_tensor = t.LongTensor(pretrained_model.tokenizer(text=in_data["text"])["input_ids"])
    	if in_tensor.shape[0] > 1024: # TODO currently, this is just throwing away perfectly good data
    	    in_tensor = in_tensor[:1024]
    	text.append(pretrained_model.tokenizer.batch_decode(in_tensor))
    	text_starts.append(text_starts[-1]+in_tensor.shape[0])
    	activations.append(autoencoder.get_features(pretrained_model.get_mlp_activations(
    	in_tensor.unsqueeze(0).to(device)).squeeze())) # A tensor of shape [input_len, autoencoder_hidden_dim]
    activations_tensor = t.cat(activations, dim=0)
    print(f"Completed {sum(len(l) for l in text)} tokens of data gathering!\nYou can now explore the activations via command line (note that this interface is temporary and janky).\n{HELP_MESSAGE}")
    while True:
        user_input = input(">>> ").split()
        if user_input[0] == "EXIT":
            break
        elif user_input[0] == "HELP":
            print(HELP_MESSAGE)
        elif user_input[0] == "STRING":
            if len(user_input) == 2:
                print(''.join(text[int(user_input[1])]))
            elif len(user_input) == 4:
                print(''.join(text[int(user_input[1])][int(user_input[2]):int(user_input[3])]))
            else:
                print("Incorrect number of arguments! Should have 1 or 3 integers")
        elif user_input[0] == "FEATURE":
            if len(user_input) == 2:
                out_acts = get_max_acts(activations_tensor, int(user_input[1]), 1)
            elif len(user_input) == 3:
                out_acts = get_max_acts(activations_tensor, int(user_input[1]), int(user_input[2]))
            else:
                print("Incorrect number of arguments! Should have 1 or 2 integers")
                break
            idxs = out_acts.indices # This is a tensor of integer indices
            for i, idx in enumerate(idxs):
                str_idx, token_idx = find_str_idx(text_starts, idx)
                string = text[str_idx]
                start_idx = max(0, token_idx-CONTEXT_VIEW_SIZE)
                end_idx = min(len(string)-1, token_idx+CONTEXT_VIEW_SIZE)
                str_to_display = string[start_idx:token_idx] + [HIGHLIGHT_COLOR, string[token_idx], HIGHLIGHT_END]
                if token_idx != end_idx: str_to_display.extend(string[token_idx+1:end_idx])
                str_to_display = ''.join(str_to_display)
                print(f"Activation #{i}: String #{str_idx}, token #{token_idx}. Context:\n{str_to_display}\n")
        else:
            print("Command not recognized. Try one of EXIT, STRING, FEATURE")
        print()

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
