import numpy as np
import random
from contextlib import contextmanager

import pandas as pd
import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, input_dim, out_dim, layers=3, residual=0, dropout=None, activation=nn.ReLU):
        super().__init__()

        step = int(round((out_dim - input_dim) / layers))
        self.residual = residual

        modules = []
        for _ in range(layers - 1):
            modules.append(nn.Linear(input_dim, input_dim + step))
            modules.append(activation(inplace=True))
            if dropout:
                modules.append(nn.Dropout(dropout))
            input_dim = input_dim + step
        modules.append(nn.Linear(input_dim, out_dim))
        self.module = nn.Sequential(*modules)

    def forward(self, input):
        output = self.module(input)
        return self.residual * input + (1 - self.residual) * output if self.residual > 0 else output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_train_samples=10, n_test_samples=100):
        super(RNN, self).__init__()
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.n_samples = self.n_train_samples

        self.RNN = torch.nn.LSTM(input_size, hidden_size)
        self.bayes_MLP = DenseBlock(hidden_size, 1, dropout=0.25)

    def forward(self, input):
        _, (hx, cx) = self.RNN(input)

        samples = []
        for _ in range(self.n_samples):
            samples.append(self.bayes_MLP(hx.squeeze()))
        samples_tensor = torch.stack(samples, dim=0).squeeze()
        return (samples_tensor.mean(0), samples_tensor.std(0))

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    @contextmanager
    def eval_mode(self):
        """Switch to evaluation mode with MC Dropout active."""
        istrain_RNN = self.RNN.training
        istrain_bayes_MLP = self.bayes_MLP.training
        try:
            self.n_samples = self.n_test_samples
            self.RNN.eval()
            self.bayes_MLP.eval()
            # Keep dropout active
            for m in self.bayes_MLP.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
            yield self.RNN, self.bayes_MLP, self.n_samples
        finally:
            self.n_samples = self.n_train_samples
            if istrain_RNN:
                self.RNN.train()
            if istrain_bayes_MLP:
                self.bayes_MLP.train()


class Trainer:
    def __init__(self, input_size, hidden_size, device):
        self.epochs = 10000
        self.device = device
        self.model = RNN(input_size, hidden_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def learn(self, train_dataloader, val_dataloader):
        self.model.train()
        for epoch in range(self.epochs):
            train_loss = self.train_step(train_dataloader)
            eval_loss = self.eval_step(val_dataloader)
            print(f"[Epoch {epoch:.0f}] Train loss: {train_loss:.2f} || Eval. Loss: {eval_loss:.2f}")
        return

    def train_step(self, train_dataloader):
        loss_epoch = 0
        seqs, labels = train_dataloader
        for seq, label in zip(seqs, labels):
            output_mean, std = self.model(seq.to(self.device))
            loss = self.criterion(output_mean, label.squeeze().to(self.device))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        return loss_epoch / len(seqs)

    @torch.inference_mode()
    def eval_step(self, val_dataloader):
        with self.model.eval_mode():
            loss_epoch = 0
            seqs, labels = val_dataloader
            for seq, label in zip(seqs, labels):
                output_mean, std = self.model(seq.to(self.device))
                loss = self.criterion(output_mean, label.to(self.device))
                loss_epoch += loss.item()
        return loss_epoch / len(seqs)


def load_data(path, window=100, batch_size=64):
    # Open, high, low, close, adj close, volume
    ohlcav_array = np.array(pd.read_csv(path).to_numpy()[:, 1:], dtype=float)

    seqs = []
    labels = []
    batch_seq = []
    batch_labels = []
    for i in range(ohlcav_array.shape[0]-window-1):
        seqs.append(torch.Tensor(ohlcav_array[i:i + window]))
        next_day_mean = 0.5 * (ohlcav_array[i + window + 1, 0] + ohlcav_array[i + window + 1, 3])
        labels.append(torch.Tensor([next_day_mean]))

    seqs_tensor = torch.stack(seqs, dim=1)
    labels_tensor = torch.cat(labels)
    idx = torch.randperm(labels_tensor.shape[0]).split(batch_size)
    for batch_idx in idx:
        batch_seq.append(seqs_tensor[:, batch_idx, :])
        batch_labels.append(labels_tensor[batch_idx])

    return batch_seq, batch_labels

def train_test_split(batch_seq, batch_labels):
    n_train = int(0.8*len(batch_labels))
    n_test = len(batch_labels) - n_train
    # train_idx, test_idx = torch.randperm(len(batch_seq)).split([n_train, n_test])
    # train_data = batch_seq[train_idx], batch_labels[train_idx]
    # test_data = batch_seq[test_idx], batch_labels[test_idx]

    train_data = batch_seq[:n_train], batch_labels[:n_train]
    test_data = batch_seq[-n_test:], batch_labels[-n_test:]
    return train_data, test_data

if __name__ == '__main__':
    path = 'gym_anytrading/datasets/data/STOCKS_GOOGL.csv'
    batch_seq, batch_labels = load_data(path)
    train_data, test_data = train_test_split(batch_seq, batch_labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Trainer(input_size=6, hidden_size=128, device=device).learn(train_data, test_data)
