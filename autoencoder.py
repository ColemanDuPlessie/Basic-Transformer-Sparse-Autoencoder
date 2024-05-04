from typing import Optional, Tuple
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from transformer import deepspeed_config_location, device
from transformer.model import TinyTransformer

class TinyAutoencoder(nn.Module):
    def __init__(
        self,
        in_size: int, hidden_size: int,
        pretrained_load_path: Optional[str] = None,
    ):
        super().__init__()

        self.in_layer = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(hidden_size, in_size)

        if pretrained_load_path is not None:
            self.load_pretrained(pretrained_load_path)

    def forward(self, x) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        in_data = x.reshape(-1, x.shape[-1])
        with_bias = in_data + self.out_layer.bias
        out = self.out_layer(self.relu(self.in_layer(with_bias)))
        out = out.reshape_as(x)
        loss = F.mse_loss(out, x)
        return loss, out
       
    def get_features(self, x):
        return self.in_layer(x)

    def load_pretrained(self, path: str = "model/pytorch_model.bin") -> None:
        self.load_state_dict(t.load(path, map_location=device))

if __name__ == "__main__":
    test_model = TinyAutoencoder(10, 20)
    input_ids = t.rand(8, 1, 16, 10)
    loss, out = test_model(input_ids)
    print("Done")
