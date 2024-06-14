# Based on:
# https://github.com/hkproj/pytorch-transformer
# https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_10_3_transformer_timeseries.ipynb


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy as dc
from data.prepare_dataset import load_dataset, load_dataset_most_correlation
from utils import save_fig_loss, save_fig


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
        h: The number of attention heads in the multi-head self-attention mechanism.
        dropout: The dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divided by heads"

        self.head_dim = d_model // h  # also called d_k in paper
        assert h * self.head_dim == d_model, "Embed size have to be equal to this multiplication"

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq, TODO check bias = False
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk, TODO check bias = False
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv, TODO check bias = False

        self.w_o = nn.Linear(self.head_dim * self.h, d_model, bias=False)  # Wo, in paper it is d_v * h, d_v == d_k == head_dim,
        # self.head_dim * self.h should be == d_model
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        head_dim = query.shape[-1]

        # (batch, h, seq_len, head_dim) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)  # @: matrix multiplication
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

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

        x, self.attention_scores = MultiHeadAttentionBlock.attention(q_prim, k_prim, v_prim, mask, self.dropout)

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
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=512, nhead=4, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.encoder = nn.Linear(input_dim, d_model)
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
    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, None)
        x = x.mean(dim=1)
        x = self.projection(x)
        return x


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
    avg_loss_across_batches = 0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output.view(-1), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_index % 10 == 9:  # print every 100 batches
            avg_loss_across_batches = running_loss / (batch_index+1)
            print('Batch {0}, Loss: {1:.5f}'.format(batch_index + 1, avg_loss_across_batches))

    scheduler.step()
    return avg_loss_across_batches


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
    print('Validation Loss: {0:.5f}'.format(avg_loss_across_batches))
    print('***************************************************')
    return avg_loss_across_batches


def run_model(train_dataset, test_dataset, model, X_test, y_test, scaler, param_num, log_dir):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    train_loss = []
    val_loss = []
    i = 0
    for epoch in range(num_epochs):
        train_loss.append(train_one_epoch(model, optimizer, scheduler, train_loader, epoch))
        loss = validate_one_epoch(model, test_loader)
        val_loss.append(loss)
        if len(val_loss) >= 3:
            if loss > val_loss[-2]:
                i = i + 1
        if i >= 3:
            break

    test_predictions = []
    with torch.no_grad():
        model.eval()

        batch_s = len(X_test) // 10  # Dzielimy X_test na 10 fragmentów
        predictions = []
        for i in range(10):
            start_idx = i * batch_s
            end_idx = (i + 1) * batch_s if i < 9 else len(X_test)
            X_batch = X_test[start_idx:end_idx].to(device)
            with torch.no_grad():
                pred_batch = model(X_batch)
            predictions.append(pred_batch)
        pred = torch.cat(predictions, dim=0)

        for t in range(0, lookforward, 1):
            predictions = []
            for i in range(10):
                start_idx = i * batch_s
                end_idx = (i + 1) * batch_s if i < 9 else len(X_test)
                X_batch = X_test[start_idx:end_idx].to(device)
                with torch.no_grad():
                    pred_batch = model(X_batch)
                predictions.append(pred_batch)
            pred = torch.cat(predictions, dim=0)
            for k in range(0, lookback - 1, 1):
                X_test[:, k, :] = X_test[:, k + 1, :]
            X_test[:, -1, :] = pred
        loss = loss_function(pred.cpu(), y_test)
        print(f"Test loss: {loss:.5f}")
        test_predictions = pred.detach().cpu().numpy().flatten()
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

    test_predictions = test_predictions[:-lookforward]
    new_y_test = new_y_test[lookforward:]

    test_predictions = test_predictions[:200]
    new_y_test = new_y_test[:200]
    save_fig(train_loss[1:], val_loss[1:], new_y_test, test_predictions, loss, log_dir)
    return loss

# Load data and run the model
def run():
    torch.cuda.empty_cache()
    data = load_dataset_most_correlation('../data/', percent_params)
    data['Time'] = pd.to_datetime(data['Time'])
    parameters_num = data.shape[1] - 1
    dir = f"trans_test1/trans_back{lookback}_forward{lookforward}_param{parameters_num}"

    shifted_df = prepare_dataframe_for_transformer(data, lookback)
    shifted = shifted_df.copy()
    shifted_df_as_np = shifted.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X_train, y_train, X_test, y_test = split_prepare_date(shifted_df_as_np, 0.8, parameters_num)
    train_data = list(zip(X_train, y_train))
    np.random.shuffle(train_data)
    X_train, y_train = zip(*train_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    model = TransformerModel(input_dim=1, d_model=1024, nhead=6, num_encoder_layers=6, dim_feedforward=1024, dropout=0.05)
    model.to(device)

    return run_model(train_dataset, test_dataset, model, X_test, y_test, scaler, parameters_num, log_dir=dir)


if __name__ == "__main__":
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lookback = 24
    lookforward = 1
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 50
    percent_params = 0.20
    loss_function = nn.MSELoss()

    '''loss_list = []
    loss_list.append(999999999)
    value = 0
    for i in [0.1, 0.15]:
        print(f"Percentage of parameters: {i}")
        percent_params = i
        loss = run()
        if loss < min(loss_list):
            value = i
        loss_list.append(loss)
    percent_params = value
    print(f"Best percentage of parameters: {percent_params}")

    loss_list = []
    loss_list.append(999999999)
    value = 0
    for i in [5, 10, 20, 30]:
        print(f"Lookback: {i}")
        lookback = i
        loss = run()
        if loss < min(loss_list):
            value = i
        loss_list.append(loss)
    lookback = value
    print(f"Best lookback: {percent_params}")'''

    loss_list = []
    value = 0
    for i in [1, 6, 12, 24]:
        lookforward = i
        print(f"Foorward: {i}")
        loss = run()

