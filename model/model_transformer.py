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
num_epochs = 15
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
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply the sin/cos to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # become a tensor with size (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)  # requires_grad_ -> this tensor will not be learn
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
        self.residual_connections_1 = ResidualConnection(dropout) # Sequential.modules
        self.residual_connections_2 = ResidualConnection(dropout)  # Sequential.modules

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


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Create custom encoder layers
        encoder_blocks = []
        for _ in range(num_encoder_layers):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, nhead, dropout)
            feed_forward_block = FeedForwardBlock(d_model, dim_feedforward, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        self.transformer_encoder = Encoder(nn.ModuleList(encoder_blocks))
        self.d_model = d_model
        self.projection = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)  # Here `None` is used as mask
        output = output.mean(dim=1)
        output = self.projection(output)
        return output


def build_transformer(d_model: int = 512, N: int = 2, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048, seq_len: int = 100) -> TransformerModel:
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
    # Create the transformer
    model = TransformerModel(d_model=d_model, nhead=h, num_encoder_layers=N, dim_feedforward=d_ff, dropout=dropout)
    return model


# Define helper functions and dataset preparation
def prepare_dataframe_for_transformer(df, n_steps):
    df.set_index('Time', inplace=True)
    data = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == "Energy":
            data[col] = df[col]
        for j in range(1, n_steps + 1):
            data[f'{col}(t-{j})'] = df[col].shift(j)
    cols = ['Energy'] + [col for col in data.columns if col != 'Energy']
    data = data[cols]
    data.dropna(inplace=True)
    return data


def split_prepare_date(shifted_df_as_np, ratio=0.95, param_num=1):
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = dc(np.flip(X, axis=1))
    split_index = int(len(X) * ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    X_train = X_train.reshape((-1, lookback * param_num, 1))
    X_test = X_test.reshape((-1, lookback * param_num, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    return X_train, y_train, X_test, y_test


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Define training and evaluation functions
def train_one_epoch(model, optimizer, scheduler, train_loader, epoch):
    model.train()
    current_lr = scheduler.get_last_lr()
    print(f'Epoch {epoch + 1}, Current Learning Rate: {current_lr}')
    running_loss = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output.view(-1), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.5f}'.format(batch_index + 1, avg_loss_across_batches))
            running_loss = 0.0

    scheduler.step()
    print()


def validate_one_epoch(model, test_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output = model(x_batch)
            loss = loss_function(output.view(-1), y_batch.view(-1))
            running_loss += loss.item()
    avg_loss_across_batches = running_loss / len(test_loader)
    print('Validation Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


def run_model(train_dataset, test_dataset, model, X_test, y_test, scaler, param_num):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, scheduler, train_loader, epoch)
        validate_one_epoch(model, test_loader)

    test_predictions = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(X_test), batch_size):
            batch_X_test = X_test[i:i + batch_size].to(device)
            batch_predictions = model(batch_X_test).detach().cpu().numpy().flatten()
            test_predictions.extend(batch_predictions)
            torch.cuda.empty_cache()

    test_predictions = np.array(test_predictions)

    dummies = np.zeros((X_test.shape[0], lookback * param_num + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X_test.shape[0], lookback * param_num + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])

    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

# Load data and run the model
def run():
    torch.cuda.empty_cache()
    data = load_dataset_most_correlation('../data/', 0.5)
    data['Time'] = pd.to_datetime(data['Time'])
    data.drop(['Soil Temperature_7-28 cm down[°C]', 'Soil Moisture_7-28 cm down[m³/m³]', 'Snow Depth_sfc[m]', 'Shortwave Radiation_sfc[W/m²]'], axis=1, inplace=True)
    parameters_num = data.shape[1] - 1

    shifted_df = prepare_dataframe_for_transformer(data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X_train, y_train, X_test, y_test = split_prepare_date(shifted_df_as_np, 0.95, parameters_num)
    train_data = list(zip(X_train, y_train))
    np.random.shuffle(train_data)
    X_train, y_train = zip(*train_data)
    X_train = np.array(X_train, dtype=object)
    y_train = np.array(y_train, dtype=object)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    model = TransformerModel(d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1).to(device)
    model.to(device)

    run_model(train_dataset, test_dataset, model, X_test, y_test, scaler, parameters_num)


if __name__ == "__main__":
    run()
