import subprocess

import wandb
from transformers import TrainerCallback


def get_nvidia_smi_output():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.stdout
    except:
        return "Failed to run nvidia-smi"


class NvidiaSMILogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # If step is a multiple of 500, log nvidia-smi output to wandb
        if state.global_step % 500 == 0:
            nvidia_output = get_nvidia_smi_output()

            wandb.log({"nvidia-smi": wandb.Html(nvidia_output)})
