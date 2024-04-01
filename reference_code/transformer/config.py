from dataclasses import asdict, dataclass

from transformer import debug, artifacts_prefix


@dataclass
class TinyConfig:
    # model hyperparameters
    num_layers: int = 1
    num_heads: int = 8
    hidden_dim: int = 128
    hidden_size: int = 128
    dropout: float = 0.1
    expert_dropout: float = 0.25
    mult: int = 4
    tokenizer_str: str = "gpt2"
    vocab_size: int = 50257

    # training hyperparameters
    batch_size: int = 2 if debug else 64
    max_seq_len: int = 1024
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    num_epochs: int = 128
    num_warmup_steps: int = 0
    weight_decay: float = 0.01
    data_str: str = "simple"
    use_combined_dataset: bool = True
    use_moe: bool = True

    # other
    def to_dict(self):
        return asdict(self)


MAX_TOKEN_LENGTH = 60


@dataclass
class ModelParts:
    config: TinyConfig
    file_name: str


# DO NOT CHANGE
TRANSFORMER_MODELS = {
    "main_transformer": ModelParts(
        config=TinyConfig(use_moe=False),
        file_name=artifacts_prefix + "artifacts/transformer/checkpoint-28000/pytorch_model.bin",
    ),
    "end_state_transformer": ModelParts(
        config=TinyConfig(use_moe=False),
        file_name=artifacts_prefix + "artifacts/transformer/end_state_model/pytorch_model.bin",
    ),
    "en_fr_moe": ModelParts(
        config=TinyConfig(use_moe=True),
        file_name=artifacts_prefix + "artifacts/moe_en_fr/pytorch_model.bin",
    ),
    "en_code_moe": ModelParts(
        config=TinyConfig(use_moe=True),
        file_name=artifacts_prefix + "artifacts/moe_en_code/pytorch_model.bin",
    ),
}
