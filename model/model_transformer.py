# Based on:
# https://github.com/hkproj/pytorch-transformer
# https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_10_3_transformer_timeseries.ipynb


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy as dc
from data.prepare_dataset import load_dataset, load_dataset_most_correlation

# Config
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
lookback = 10
batch_size = 32
learning_rate = 0.001
num_epochs = 3
loss_function = nn.MSELoss()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int = 5000) -> None:
        """
        seq_len: maximum length of sentence
        dropout: to make model less overfit
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply the sin/cos to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe = pe.unsqueeze(0).transpose(0, 1)

        pe = pe.unsqueeze(0)  # become a tensor with size (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # requires_grad_ -> this tensor will not be learn
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)  # standard deviation
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        d_model: The number of features in the transformer model's internal representations (also the size of
        embeddings). This controls how much a model can remember and process.
        nhead: The number of attention heads in the multi-head self-attention mechanism.
        dropout: The dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divided by heads"

        self.head_dim = d_model // h  # also called d_k in paper
        assert h * self.head_dim == d_model, "Embed size have to be equal to this multiplication"

        self.w_q = nn.Linear(d_model, d_model)  # Wq, TODO check bias = False
        self.w_k = nn.Linear(d_model, d_model)  # Wk, TODO check bias = False
        self.w_v = nn.Linear(d_model, d_model)  # Wv, TODO check bias = False

        self.w_o = nn.Linear(self.head_dim * self.h, d_model)  # Wo, in paper it is d_v * h, d_v == d_k == head_dim,
        # self.head_dim * self.h should be == d_model
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dim = query.shape[-1]

        # (batch, h, seq_len, head_dim) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)  # @: matrix multiplication

        # if mask is not None:
        #     h = attention_scores.size(1)
        #     mask = mask.unsqueeze(1)
        #     mask = mask.expand(-1, h, -1, -1)
        #     attention_scores = attention_scores.masked_fill(mask == 0, float('-1e20'))
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value)

    def forward(self, q, k, v, mask):
        q_prim = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        k_prim = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        v_prim = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # divide to smaller matrix's (batch, seq_len, d_model) --> (batch, seq_len, h, head_dim),
        # embeddings split into edge parts (batch [0], seq [1] is not  splited), -->
        # -->{transpose}-> (batch, h, seq_len, head_dim)
        q_prim = q_prim.view(q_prim.shape[0], q_prim.shape[1], self.h, self.head_dim).transpose(1, 2)
        k_prim = k_prim.view(k_prim.shape[0], k_prim.shape[1], self.h, self.head_dim).transpose(1, 2)
        v_prim = v_prim.view(v_prim.shape[0], v_prim.shape[1], self.h, self.head_dim).transpose(1, 2)

        x = MultiHeadAttentionBlock.attention(q_prim, k_prim, v_prim, mask, self.dropout)

        # (batch, h, seq_len, head_dim) --> (batch, seq_len, h, head_dim) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.head_dim)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections_1 = nn.Sequential(ResidualConnection(dropout)) # Sequential.modules
        self.residual_connections_2 = nn.Sequential(ResidualConnection(dropout)) # Sequential.modules

    def forward(self, x, src_mask):
        x = self.residual_connections_1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections_2(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Last layer used for predicting
    """

    def __init__(self, d_model, output_size=1) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, output_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, output_size)
        return self.projection(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, src_pos: PositionalEncoding, projection_layer: ProjectionLayer,
                 heads_num) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_pos = src_pos
        self.src_mask = None
        self.projection_layer = projection_layer
        self.h = heads_num

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def project(self, x):
        # (batch, seq_len, output_size)
        return self.projection_layer(x)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(2) != len(src):
            device = src.device
            batch_size = src.size(0)
            sz = src.size(1)
            self.src_mask = self._generate_square_subsequent_mask(sz, batch_size).to(device)

        encoder_output = self.encode(src, self.src_mask)
        return self.project(encoder_output)

    def _generate_square_subsequent_mask(self, sz, bt_size):
        mask = torch.triu(torch.ones(bt_size, sz, sz) * float('-inf'), diagonal=1)
        return mask


def build_transformer(d_model: int = 512, N: int = 2, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048, seq_len: int = 100) -> Transformer:
    """
    Args:
        d_model:
        N: num of encoder block
        h: num of heads
        dropout: droput probability
        d_ff: hidden layer [FF] size
        seq_len:
    Returns:

    """
    # Create the positional encoding layer
    src_pos = PositionalEncoding(d_model, dropout, seq_len)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model)

    # Create the transformer
    transformer = Transformer(encoder, src_pos, projection_layer, h)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))

    return torch.FloatTensor(inout_seq)


def test_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_window = 10
    output_window = 1
    L = 1790  # Długość sztucznego szeregu czasowego
    data = np.sin(np.linspace(0, 100, L)) + np.random.normal(0, 0.1, (L,))

    train_data = create_inout_sequences(data, input_window, output_window)

    d_model = 512  # Wymiar ukryty modelu
    N = 2  # Liczba bloków w encoderze
    h = 8  # Liczba głów w MultiHeadAttention
    dropout = 0.1
    d_ff = 2048  # Wymiar warstwy FeedForward
    batch_size = 250
    transformer = build_transformer(d_model, N, h, dropout, d_ff, seq_len=input_window)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    epochs = 5

    for epoch in range(1, epochs + 1):
        transformer.train()
        total_loss = 0.
        for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
            data, targets = get_batch(train_data, i, batch_size, input_window)
            optimizer.zero_grad()
            output = transformer(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch}, Loss: {total_loss / len(train_data)}')


if __name__ == "__main__":
    test_transformer()
