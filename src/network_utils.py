import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

def generate_scaled_var_drop_mask(shape, keep_prob):
    assert keep_prob > 0. and keep_prob <= 1.
    mask = torch.rand(shape, device='cuda').le(keep_prob)
    mask = mask.float() / keep_prob
    return mask

class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

class BERTLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertFeedForward(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super(BertFeedForward, self).__init__()
        self.dense = nn.Linear(input_size, intermediate_size)
        self.affine = nn.Linear(intermediate_size, output_size)
        self.act_fn = gelu
        self.LayerNorm = BERTLayerNorm(intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.affine(hidden_states)
        return hidden_states
