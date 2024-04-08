from typing import Optional, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.transformer import Transformer
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)
from transformer import device

from transformer.config import TinyConfig

if t.cuda.is_available():
    current_device = t.cuda.current_device()
    device_name = t.cuda.get_device_name(current_device)


class TinyTransformer(PreTrainedModel):
    def __init__(
        self,
        model_config: TinyConfig = TinyConfig(),
        config: PretrainedConfig = PretrainedConfig(),
        pretrained_load_path: Optional[str] = None,
    ):
        config.hidden_size = model_config.hidden_dim
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_str)
        self.tokenizer.pad_token_id = [self.tokenizer.eos_token_id]
        self.model_config = model_config

        self.vocab_size = len(self.tokenizer)

        self.token_embedding = nn.Embedding(self.vocab_size, model_config.hidden_dim)
        self.position_embedding = nn.Embedding(
            model_config.max_seq_len, model_config.hidden_dim
        )
        self.unembedding = nn.Linear(model_config.hidden_dim, self.vocab_size)

        self.mlp = nn.Sequential(
            nn.Linear(
                model_config.hidden_dim, model_config.hidden_dim * model_config.mult
            ),
            nn.ReLU(),
            nn.Linear(
                model_config.hidden_dim * model_config.mult, model_config.hidden_dim
            ),
        )

        self.attn = nn.MultiheadAttention(
            model_config.hidden_dim,
            model_config.num_heads,
            dropout=model_config.dropout,
            batch_first=True,
        )

        self.W_qkv = nn.Linear(model_config.hidden_dim, model_config.hidden_dim * 3)
        self.W_o = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)

        self.dropout = nn.Dropout(model_config.dropout)

        self.ln1 = nn.LayerNorm(model_config.hidden_dim)
        self.ln2 = nn.LayerNorm(model_config.hidden_dim)

        self.final_layer_norm = nn.LayerNorm(model_config.hidden_dim)

        if pretrained_load_path is not None:
            self.load_pretrained(pretrained_load_path)

    def forward(self, input_ids: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        batch_size, seq_len = input_ids.shape

        # Get the embeddings
        embed = self.token_embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        # print("input_shape", input_ids.shape)
        pos = self.position_embedding(
            t.arange(input_ids.size(1), device=input_ids.device)
        )  # (batch_size, seq_len, hidden_dim)
        x: t.Tensor = self.dropout(embed + pos)  # (batch_size, seq_len, hidden_dim)

        # Attention
        x = self.ln1(x)
        qkv = self.W_qkv(x)  # (batch_size, seq_len, hidden_dim * 3)
        q, k, v = t.split(
            qkv, self.model_config.hidden_dim, dim=2
        )  # (batch_size, seq_len, hidden_dim)
        attention_mask = (
            Transformer.generate_square_subsequent_mask(seq_len)
            .to(x.device)
            .to(x.dtype)
        )

        y, _ = self.attn(
            query=q, key=k, value=v, is_causal=True, attn_mask=attention_mask
        )  # (batch_size, seq_len, hidden_dim)
        y = self.W_o(y)  # (batch_size, seq_len, hidden_dim)
        x = x + y
        x = self.dropout(x)

        # MLP
        x = self.ln2(x)
        y = self.mlp(x)  # (batch_size, seq_len, hidden_dim)
        x = x + y

        x = self.final_layer_norm(x)

        logits: t.Tensor = self.unembedding(x)  # (batch_size, seq_len, vocab_size)

        flattened_logits = rearrange(logits[:, :-1, :], "b s v -> (b s) v")
        flattened_labels = rearrange(input_ids[:, 1:], "b s -> (b s)")

        loss = F.cross_entropy(flattened_logits, flattened_labels, reduction="mean")

        return loss, logits
    
    def get_mlp_activations(self, input_ids: t.Tensor) -> t.Tensor:
        batch_size, seq_len = input_ids.shape

        # Get the embeddings
        embed = self.token_embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        # print("input_shape", input_ids.shape)
        pos = self.position_embedding(
            t.arange(input_ids.size(1), device=input_ids.device)
        )  # (batch_size, seq_len, hidden_dim)
        x: t.Tensor = self.dropout(embed + pos)  # (batch_size, seq_len, hidden_dim)

        # Attention
        x = self.ln1(x)
        qkv = self.W_qkv(x)  # (batch_size, seq_len, hidden_dim * 3)
        q, k, v = t.split(
            qkv, self.model_config.hidden_dim, dim=2
        )  # (batch_size, seq_len, hidden_dim)
        attention_mask = (
            Transformer.generate_square_subsequent_mask(seq_len)
            .to(x.device)
            .to(x.dtype)
        )

        y, _ = self.attn(
            query=q, key=k, value=v, is_causal=True, attn_mask=attention_mask
        )  # (batch_size, seq_len, hidden_dim)
        y = self.W_o(y)  # (batch_size, seq_len, hidden_dim)
        x = x + y
        x = self.dropout(x)

        # MLP
        x = self.ln2(x)
        
        for layer in self.mlp[:-1]:
            x = layer(x)
        return x

    def load_pretrained(self, path: str = "model/pytorch_model.bin") -> None:
        self.load_state_dict(t.load(path, map_location=device))


if __name__ == "__main__":
    model = TinyTransformer()
    input_ids = t.randint(0, 100, (10, 10))
    loss, logits = model(input_ids)
    print("Done")
