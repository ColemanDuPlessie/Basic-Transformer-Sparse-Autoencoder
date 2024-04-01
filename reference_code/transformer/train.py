import argparse
import os
import sys
from typing import Dict, Optional, Tuple, Union

import torch as t
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import Trainer, TrainingArguments

import wandb
from datasets import Dataset
from transformer import debug, deepspeed_config_location, device
from transformer.combined_dataset import (
    get_combined_dataset,
    get_en_code_dataset,
    get_en_fr_dataset,
)
from transformer.config import TinyConfig
from transformer.data_prep import (
    data_collator,
    get_individual_test_dataset,
    get_individual_train_dataset,
)
from transformer.model import TinyTransformer
from transformer.moe import TinyMoE
from transformer.utils import NvidiaSMILogger
from wandb import AlertLevel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

DATA_SETS: Dict[str, Tuple[str, str, int]] = {
    "simple": ("wikipedia", "20220301.simple", 400),
    "english": ("wikipedia", "20220301.en", 4000),
}

GRADIENT_ACCUMULATION_STEPS = 4

print(f"Torch version: {t.__version__}")
print(f"Using device {device}")


def train(
    model: Union[TinyMoE, TinyTransformer],
    model_config: TinyConfig,
    train_dataset: TorchIterableDataset,
    test_dataset: Dataset,
    num_steps: int,
    debug: bool,
    name: str,
):
    tokenizer = model.tokenizer

    checkpoint_dir = "./results/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        max_steps=20 if debug else num_steps,
        num_train_epochs=1,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,
        warmup_steps=100 if device == "cuda" else 0,
        learning_rate=model_config.learning_rate if device == "cuda" else None,
        weight_decay=model_config.weight_decay,
        save_steps=100,
        save_total_limit=2,
        logging_steps=5 if (debug or GRADIENT_ACCUMULATION_STEPS > 1) else 10,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # eval_steps=1000,
        deepspeed=deepspeed_config_location,
        half_precision_backend="deepspeed" if device == "cuda" else "auto",
        fp16=True if device == "cuda" else False,
        # load_best_model_at_end=True,
    )

    # TODO: Improve utilisation/throughput

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[NvidiaSMILogger()],
    )

    wandb_run = wandb.init(
        project="dictionary_learning",
        name=name,
        config=model_config.to_dict(),
    )

    with wandb_run:  # type: ignore
        try:
            wandb.alert(title="Training started", text="Training started")
            trainer.train()
            wandb.alert(title="Training complete", text="Training complete")
        except Exception as e:
            wandb.alert(
                title="Training failed",
                text="Training failed",
                level=AlertLevel.ERROR,
            )
            raise e

        # Save model
        save_dir = f"./model/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_location = f"{save_dir}/{name}"
        trainer.save_model(save_location)
        wandb.save(f".{save_location}/*")
        wandb.finish()

    # Test the model with predictions

    first_five_inputs = test_dataset[:5]["input_ids"]
    print("Input:", first_five_inputs)
    # print("Input:", tokenizer.decode_batch(first_five_inputs))

    for i, input in enumerate(first_five_inputs):
        input: t.Tensor
        _, logits = model(
            input.unsqueeze(0).to(device)
        )  # (batch_size, seq_len, vocab_size)
        out = t.argmax(logits, dim=-1)  # (batch_size, seq_len)
        str_out = tokenizer.decode(out.squeeze(0).tolist())
        print(f"Output {i}:", str_out)


def main(debug: bool, name: str, data_str: Optional[str] = None):
    ...
    model_config: TinyConfig = TinyConfig()
    if model_config.use_moe:
        model = TinyMoE()
    else:
        model = TinyTransformer()
    model.to(device)

    print("model", model)

    # def print_grad_hook(module, grad_input, grad_output):
    #     print(f"Inside '{module.__class__.__name__}' backward")
    #     for i, g in enumerate(grad_input):
    #         print(f"Grad input {i}: {g}")
    #     for i, g in enumerate(grad_output):
    #         print(f"Grad output {i}: {g}")

    # model.lin_router.register_full_backward_hook(print_grad_hook)
    # model.lin_router = model.lin_router.float()

    # print("Added backward hook")

    DATA_PATH, DATA_NAME, DATA_NUM_STEPS = DATA_SETS[model_config.data_str]

    # ray.init(num_cpus=60 if not debug else 2)
    # Get streaming train dataset
    if data_str:
        if data_str == "en_code":
            train_dataset, DATA_NUM_STEPS = get_en_code_dataset()
            print("Using en_code dataset")
        elif data_str == "en_fr":
            train_dataset, DATA_NUM_STEPS = get_en_fr_dataset()
            print("Using en_fr dataset")
        else:
            raise ValueError(
                f"Unknown data_str: {data_str}, please enter either 'en_code' or 'en_fr'"
            )
    elif model_config.use_combined_dataset:
        train_dataset, DATA_NUM_STEPS = get_combined_dataset()
    else:
        train_dataset = get_individual_train_dataset(
            DATA_PATH, DATA_NAME, DATA_NUM_STEPS
        )

    # Get static test dataset
    test_dataset = get_individual_test_dataset(DATA_PATH, DATA_NAME)

    train(
        model=model,
        model_config=model_config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_steps=DATA_NUM_STEPS
        // GRADIENT_ACCUMULATION_STEPS,  # gradient accumulation reduces the number of steps
        debug=debug,
        name=name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with optional GPU support."
    )
    # parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gpu", type=int, help="Use GPU for training")
    parser.add_argument(
        "--data", type=str, help="Data to use for training", default=None
    )
    args = parser.parse_args()

    local_gpu_rank = int(args.gpu) if args.gpu else None
    if args.data:
        data_str = args.data
        name = f"moe_transformer_{data_str}"
    else:
        data_str = "en_fr"
        name = f"moe_transformer"

    main(debug=debug, name=name, data_str=data_str)
    # ray.shutdown()
