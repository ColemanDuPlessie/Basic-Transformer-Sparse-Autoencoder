from typing import Optional, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from torch.nn import ModuleList
from torch.nn.modules.transformer import Transformer
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from transformer import device
from transformer.config import TinyConfig

if t.cuda.is_available():
    current_device = t.cuda.current_device()
    device_name = t.cuda.get_device_name(current_device)

NUM_EXPERTS = 2


def load_balancing_aux_loss_function(router_logits: t.Tensor) -> t.Tensor:
    batch_size, seq_len, num_experts = router_logits.shape
    router_logits = rearrange(router_logits, "batch seq experts -> (batch seq) experts")

    # Collect how many tokens each expert has a higher affinity for
    perm_matrix = t.where(
        router_logits == t.max(router_logits, dim=-1, keepdim=True).values,
        t.ones_like(router_logits),
        t.zeros_like(router_logits),
    )  # [batch_seq, expert]

    total_tokens_per_expert = reduce(
        perm_matrix, "batch_seq expert -> expert", "sum"
    )  # [expert]
    frac_tokens_per_expert = total_tokens_per_expert / (seq_len * batch_size)

    routing_probs = F.softmax(router_logits, dim=-1)  # [num_experts, batch_seq]

    total_router_prob_per_expert = reduce(
        routing_probs, "batch_seq expert -> expert", "sum"
    )  # [layer, num_experts]
    frac_router_prob_per_expert = total_router_prob_per_expert / (seq_len * batch_size)

    # Dot product
    lb_loss = num_experts * einsum(
        frac_tokens_per_expert,
        frac_router_prob_per_expert,
        "expert, expert ->",
    )

    return lb_loss


def router_z_loss_function(router_logits: t.Tensor) -> t.Tensor:
    batch_size, seq_len, num_experts = router_logits.shape
    router_logits = rearrange(
        router_logits, "batch seq experts -> experts (batch seq)"
    )  # num_experts batch_seq

    lse_logits = t.logsumexp(router_logits, dim=-1)  # [num_experts]
    squared_lse_logits = lse_logits ** 2

    z_loss = einsum(squared_lse_logits, "num_experts ->") / (seq_len)

    return z_loss


class TinyMoE(PreTrainedModel):
    # def backward_hook(tensor, grad):
    #     print(f"Gradient value: {grad}")
    #     return None
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

        self.experts = ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        model_config.hidden_dim,
                        model_config.hidden_dim * model_config.mult,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        model_config.hidden_dim * model_config.mult,
                        model_config.hidden_dim,
                    ),
                )
                for _ in range(NUM_EXPERTS)
            ]
        )

        self.lin_router = nn.Linear(model_config.hidden_dim, NUM_EXPERTS
                                    # , dtype = t.float32
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

        self.z_coef = 0.1
        self.lb_coef = 0.01

        self.expert_dropout = nn.Dropout(model_config.expert_dropout)

        if pretrained_load_path is not None:
            self.load_pretrained(pretrained_load_path)

    def forward(self, input_ids: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
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

        # MoE Layer
        x = self.ln2(x)
        router_logits = self.lin_router(x)

        # print(x.float().dtype, self.lin_router.weight.dtype, self.lin_router.bias.dtype)
        # router_logits = self.lin_router(x.float())  # (batch_size, seq_len, num_experts)
        # print(f"max_router_logit: {t.max(router_logits)}, min_router_logit: {t.min(router_logits)}")
        # Add gumbel noise
        # gumbel_noise = -t.log(-t.log(t.rand_like(router_logits) + 1e-10) + 1e-10)
        # noised_router_logits = router_logits + gumbel_noise
        # router_probs = F.softmax(
        #     noised_router_logits if self.training else router_logits, dim=-1
        # )
        # TODO: Check in on gumbel softmax trick
        router_probs = F.softmax(router_logits, dim=-1)
        # Keep only the max router_prob in each row
        router_probs = t.where(
            router_probs == t.max(router_probs, dim=-1, keepdim=True).values,
            router_probs,
            t.zeros_like(router_probs),
        )

        # Doing inefficient version for time reasons
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(self.expert_dropout(expert(x)))
        expert_outputs = t.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, hidden_dim, num_experts)
        expert_outputs = expert_outputs.to(device)
        y = einsum(
            expert_outputs,
            router_probs,
            "batch seq hidden_dim expert, batch seq expert -> batch seq hidden_dim",
        )
        x = x + y

        x = self.final_layer_norm(x)

        logits: t.Tensor = self.unembedding(x)  # (batch_size, seq_len, vocab_size)

        flattened_logits = rearrange(logits[:, :-1, :], "b s v -> (b s) v")
        flattened_labels = rearrange(input_ids[:, 1:], "b s -> (b s)")

        cx_loss = F.cross_entropy(flattened_logits, flattened_labels, reduction="mean")
        router_z_loss = router_z_loss_function(router_logits)
        load_balancing_loss = load_balancing_aux_loss_function(router_logits)

        loss = (
                cx_loss + self.z_coef * router_z_loss + self.lb_coef * load_balancing_loss
        )
        # if t.isnan(loss):
        #     print(input_ids)
        #     raise ValueError("Loss is NaN")
        return loss, logits

    def load_pretrained(self, path: str = "model/pytorch_model.bin") -> None:
        self.load_state_dict(t.load(path, map_location=device))


if __name__ == "__main__":
    model = TinyMoE()
    print(model)
    input_ids = t.randint(0, 100, (10, 10))
    loss, logits = model(input_ids)
    print("Done")
