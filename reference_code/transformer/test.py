import torch as t
import torch.nn.functional as F

from auto_encoder import model_location
from transformer.model import TinyTransformer
from tqdm import tqdm as tdqm

# PROMPT = "1,2,3,4,5,6,7,8,9,"
# PROMPT = "Sally went to the store to buy some"
# PROMPT = "الصلاة على"
# PROMPT = "アメリカ合衆"
# PROMPT = "ACTGCGGCGATCTGACGTTCTCGAGCTCGATCGC"
# PROMPT = "bXkgbmFtZSBpcyBIQU5TIGFuZCBpIGFtIENPT0w="
PROMPT = "Je n'ai pas mangé"
NUM_NEW_TOKENS = 20
TEMPERATURE = 0.1

pretrained_transformer = TinyTransformer(pretrained_load_path="../artifacts/transformer/checkpoint-28000/pytorch_model.bin")
pretrained_transformer.eval()

tokenizer = pretrained_transformer.tokenizer

input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
input_tensor = t.tensor(input_ids)

# Generate output
for i in tdqm(range(NUM_NEW_TOKENS)):
    _loss, logits = pretrained_transformer(input_tensor)
    logits: t.Tensor = logits[0, -1, :] / TEMPERATURE # vocab_size
    distribution = F.softmax(logits, dim=-1) # vocab_size

    # Sample from the distribution
    next_token = t.multinomial(distribution, num_samples=1) # 1

    # Concat next token with input_ids
    input_tensor = t.cat([input_tensor, next_token.unsqueeze(-1)], dim=-1) # 1 seq_len+1

decoded_output = tokenizer.batch_decode(
    input_tensor, skip_special_tokens=True
)
print(decoded_output)
