import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from src.algorithm_utils import Algorithm, PyTorchUtils
from src.pytorchtools import EarlyStopping


class LSTMED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'LSTM-ED', num_epochs: int = 20, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 80, sequence_length: int = 20, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True, step=1):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.step = step

        self.lstmed = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame, seq=None, path=""):
        if seq == None:
            X.interpolate(inplace=True)
            X.bfill(inplace=True)
            data = X.values
            index = np.arange(0, data.shape[0] - self.sequence_length + 1, self.step, int)
            sequences = [data[i:i + self.sequence_length] for i in index]
        else:
            sequences = seq
        # sequences = seq
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)
        early_stopping = EarlyStopping()
        Loss = None

        self.lstmed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()
                Loss = loss
            print(Loss)

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())


        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame, t=None, model=None):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        index = np.arange(0, data.shape[0] - self.sequence_length + 1, self.step, int)
        sequences = [data[i:i + self.sequence_length] for i in index]
        # sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        attributions = []
        t_news = []

        def wrapper(x):
            mse = nn.MSELoss()
            l1 = nn.L1Loss()
            return l1(x, self.lstmed(x).float()).unsqueeze(0)

        for idx, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            explainer = IntegratedGradients(forward_func=wrapper)
            attribution, approximation_error = explainer.attribute(ts.float(), return_convergence_delta=True)
            # score = ((error - self.mean).t()) * ((self.cov)**(-1)) * (error - self.mean)
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())
                attributions.append(attribution.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        if self.step > int(self.sequence_length / 2):
            lattice = np.full((self.step + 1, data.shape[0]), np.nan)
        else:
            # lattice = np.full((self.sequence_length - self.step + 1, data.shape[0]), np.nan)
            lattice = np.full((int(self.sequence_length / self.step), data.shape[0]), np.nan)
        # lattice = np.full((2, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % (int(self.sequence_length / self.step)),
            i * self.step:i * self.step + self.sequence_length] = score
            # lattice[i % 2, i * int(self.sequence_length/2):i * int(self.sequence_length/2) + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            if self.step > int(self.sequence_length / 2):
                lattice = np.full((self.step + 1, X.shape[0], X.shape[1]), np.nan)
            else:
                lattice = np.full((int(self.sequence_length / self.step), X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % (int(self.sequence_length / self.step)),
                i * self.step:i * self.step + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})
            outputs = np.nanmean(lattice, axis=0)

            errors = np.concatenate(errors)
            if self.step > int(self.sequence_length / 2):
                lattice = np.full((self.step + 1, X.shape[0], X.shape[1]), np.nan)
            else:
                lattice = np.full((int(self.sequence_length / self.step), X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % (int(self.sequence_length / self.step)),
                i * self.step:i * self.step + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})
            errors = np.nanmean(lattice, axis=0)

            attributions = np.concatenate(attributions)
            if self.step > int(self.sequence_length / 2):
                lattice = np.full((self.step + 1, X.shape[0], X.shape[1]), np.nan)
            else:
                lattice = np.full((int(self.sequence_length / self.step), X.shape[0], X.shape[1]), np.nan)
            for i, attribution in enumerate(attributions):
                lattice[i % (int(self.sequence_length / self.step)),
                i * self.step:i * self.step + self.sequence_length, :] = attribution
            self.prediction_details.update({'attributions_mean': np.nanmean(lattice, axis=0).T})
            attributions = np.nanmean(lattice, axis=0)

        return scores, errors, outputs, attributions, self.mean, self.cov

    def predict_without_attributions(self, X: pd.DataFrame, t=None, model=None):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        index = np.arange(0, data.shape[0] - self.sequence_length + 1, self.step, int)
        sequences = [data[i:i + self.sequence_length] for i in index]
        # sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        t_news = []

        def wrapper(x):
            mse = nn.MSELoss()
            l1 = nn.L1Loss()
            return l1(x, self.lstmed(x).float()).unsqueeze(0)

        for idx, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            explainer = IntegratedGradients(forward_func=wrapper)
            # score = ((error - self.mean).t()) * ((self.cov)**(-1)) * (error - self.mean)
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        if self.step > int(self.sequence_length / 2):
            lattice = np.full((self.step + 1, data.shape[0]), np.nan)
        else:
            # lattice = np.full((self.sequence_length - self.step + 1, data.shape[0]), np.nan)
            lattice = np.full((int(self.sequence_length / self.step), data.shape[0]), np.nan)
        # lattice = np.full((2, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % (int(self.sequence_length / self.step)),
            i * self.step:i * self.step + self.sequence_length] = score
            # lattice[i % 2, i * int(self.sequence_length/2):i * int(self.sequence_length/2) + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            if self.step > int(self.sequence_length / 2):
                lattice = np.full((self.step + 1, X.shape[0], X.shape[1]), np.nan)
            else:
                lattice = np.full((int(self.sequence_length / self.step), X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % (int(self.sequence_length / self.step)),
                i * self.step:i * self.step + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})
            outputs = np.nanmean(lattice, axis=0)

            errors = np.concatenate(errors)
            if self.step > int(self.sequence_length / 2):
                lattice = np.full((self.step + 1, X.shape[0], X.shape[1]), np.nan)
            else:
                lattice = np.full((int(self.sequence_length / self.step), X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % (int(self.sequence_length / self.step)),
                i * self.step:i * self.step + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})
            errors = np.nanmean(lattice, axis=0)

        return scores, errors, outputs


class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
         self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        inputs = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])
            if self.training:
                o, dec_hidden = self.decoder(inputs[:, i].unsqueeze(1).float(), dec_hidden)
                # _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
                # o, dec_hidden = self.decoder(output[:, i, :].unsqueeze(1).float(), dec_hidden)
            else:
                o, dec_hidden = self.decoder(inputs[:, i].unsqueeze(1).float(), dec_hidden)
                # o, dec_hidden = self.decoder(output[:, i, :].unsqueeze(1).float(), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
