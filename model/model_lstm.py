# Based on https://colab.research.google.com/drive/1CBIdPxHn_W2ARx4VozRLIptBrXk7ZBoM?usp=sharing#scrollTo=iue5WvTxmVKB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# Config
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
lookback = 7
batch_size = 16
learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    df.set_index('date', inplace=True)

    for i in range(1, n_steps + 1):
        df[f'close(t-{i})'] = df['close'].shift(i)
    df.dropna(inplace=True)

    return df


def split_prepare_date(shifted_df_as_np, ratio=0.95):
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = dc(np.flip(X, axis=1))

    split_index = int(len(X) * ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

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


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTMCell, self).__init__()
        self.input_size = input_size  # (batch_size, sequence_length, feature_length) => feature_length
        self.hidden_size = hidden_size # (batch_size, output_size) => output_size
        self.device = device

        self.weight_fg_ih = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True).to(device)
        self.weight_fg_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True).to(device)
        self.bias_fg = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True).to(device)

        self.weight_ig_perc_ih = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True).to(device)
        self.weight_ig_perc_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True).to(device)
        self.bias_ig_perc = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True).to(device)

        self.weight_ig_ih = nn.Parameter(torch.Tensor(input_size, hidden_size)).to(device)
        self.weight_ig_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size)).to(device)
        self.bias_ig = nn.Parameter(torch.Tensor(hidden_size)).to(device)

        self.weight_og_ih = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True).to(device)
        self.weight_og_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True).to(device)
        self.bias_og = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            torch.nn.init.uniform_(w, -std, std)

    def forward(self, input, hidden):
        hx, cx = hidden

        # Forget gate (percent long term to remember)
        fg = torch.sigmoid((input * self.weight_fg_ih) + (hx * self.weight_fg_hh) + self.bias_fg)

        # Input gate perc (percent potential memory to remember)
        ig_perc = torch.sigmoid((input * self.weight_ig_ih) + (hx * self.weight_ig_hh) + self.bias_ig)

        # Input gate val (potential long-term memory)
        ig = torch.tanh((input * self.weight_ig_ih) + (hx * self.weight_ig_hh) + self.bias_ig)

        # New cell state
        updated_long_memory = ((cx * fg) + (ig * ig_perc))

        # Output gate
        og = torch.sigmoid((input * self.weight_og_ih) + (hx * self.weight_og_hh) + self.bias_og)

        # New hidden state
        updated_short_memory = og * torch.tanh(updated_long_memory)

        return updated_short_memory, (updated_short_memory, updated_long_memory)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, dev):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_stacked_layers
        self.device = dev

        self.layers = nn.ModuleList(
            [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, dev) for i in range(num_stacked_layers)])
        self.fc = nn.Linear(hidden_size, 1).to(dev)

    def forward(self, x):
        batch_size = x.size(0)
        h = [torch.zeros(batch_size, x.size(2)).to(self.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, x.size(2)).to(self.device) for _ in range(self.num_layers)]

        for i in range(x.size(1)):
            for layer in range(self.num_layers):
                output, (h[layer], c[layer])= self.layers[layer](x[:, i, :], (h[layer], c[layer]))

        last_hidden = h[-1]
        out = self.fc(last_hidden.to(self.device))
        return out.to(self.device)


# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_stacked_layers):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_stacked_layers = num_stacked_layers
#
#         self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
#                             batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
#
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out


def train_one_epoch(model, optimizer, train_loader, epoch):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()


def validate_one_epoch(model, test_loader):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


def run_model(train_dataset, test_dataset, model, X_test, y_test, scaler):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, epoch)
        validate_one_epoch(model, test_loader)

    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])

    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()


def run():
    data = pd.read_csv('data/market/000001.SZ.csv')
    data = data[['date', 'close']]
    data['date'] = pd.to_datetime(data['date'])

    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X_train, y_train, X_test, y_test = split_prepare_date(shifted_df_as_np)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    model = LSTM(input_size=1, hidden_size=1, num_stacked_layers=1, dev=device)
    model.to(device)

    run_model(train_dataset, test_dataset, model, X_test, y_test, scaler)


if __name__ == "__main__":
    run()
