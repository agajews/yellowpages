import torch
import torch.nn.functional as F
import math
from utils import resolve_activation_fn


class MLP(torch.nn.Module):
    def __init__(
        self, layer_shapes, activation="relu", bias=True, final_activation=False
    ):
        """
        a simple, generic MLP class. it has len(layer_shapes)-1 layers, each followed by an activation function.

        layer_shapes: [n_in, hidden_shapes..., n_out]
        activation: either an activation function, or the string name of an attribute of torch.nn.functional
        bias: whether or not to have a bias on the linear layers
        final_activation: whether or not to have an activation function after the last layer
        """
        super().__init__()

        self.activation = resolve_activation_fn(activation)
        self.final_activation = final_activation

        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_shapes) - 1):
            self.layers.append(
                torch.nn.Linear(layer_shapes[i], layer_shapes[i + 1], bias=bias)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.final_activation:
                x = self.activation(x)
        return x


class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq, hidden)
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """
        x: (batch, seq, hidden)
        """
        x = x + self.pos_embedding[:, : x.size(1)]
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        hidden,
        n_layers=8,
        n_heads=8,
        max_len=5000,
        positional_encoding="sin",
        causal=False,
    ):
        super().__init__()

        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=hidden * 4,
            dropout=0,
            norm_first=True,
        )
        self.model = torch.nn.TransformerEncoder(layer, num_layers=n_layers)

        if positional_encoding == "sin":
            self.positional_encoding = SinusoidalPositionalEncoding(hidden, max_len)
        elif positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(hidden, max_len)

        self.causal = causal
        self.causal_mask = self.register_buffer(
            "causal_mask",
            torch.nn.Transformer.generate_square_subsequent_mask(max_len),
        )

    def forward(self, x):
        """
        x: (batch, seq, hidden)
        """
        x = self.positional_encoding(x)
        if self.causal:
            return self.model(x, mask=self.causal_mask)
        else:
            return self.model(x)