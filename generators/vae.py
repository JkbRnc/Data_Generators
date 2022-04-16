import torch
from torch import nn
from torch.nn import functional as F

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder

class Encoder(nn.Module):
  def __init__(self, input_dim, latent_dim = 32, hidden_params = [256, 256]):
      super(Encoder, self).__init__()
      hidden_layers = []
      input = input_dim
      for layer in hidden_params:
        hidden_layers += [
                          nn.Linear(input, layer),
                          nn.Dropout(0.3),
                          nn.ReLU()
                          
        ]
        input = layer
      
      self.model = nn.Sequential(*hidden_layers)
      self.mean = nn.Linear(input, latent_dim)
      self.logvar = nn.Linear(input, latent_dim)

  def forward(self, x):
    hidden = self.model(x)
    return self.mean(hidden), self.logvar(hidden)

class Decoder(nn.Module):
  def __init__(self, output_dim, input_dim = 32, hidden_params = [256, 256]):
      super(Decoder, self).__init__()
      hidden_layers = []
      input = input_dim
      for layer in hidden_params:
        hidden_layers += [
                          nn.Linear(input, layer),
                          nn.Dropout(0.3),
                          nn.ReLU()
        ]
        input = layer
      
      hidden_layers += [nn.Linear(input, output_dim)]
      self.model = nn.Sequential(*hidden_layers)

  def forward(self, x):
    return self.model(x)

class VAE():
  def __init__(self, latent_dim = 32, e_params = [256, 256], d_params = [256, 256], num_epochs = 300, batch_size = 512, lr = 1e-3):
    self.encoder = None
    self.decoder = None
    self.continous_scaler = StandardScaler()
    self.categorical_scaler = OneHotEncoder()
    self.latent_dim = latent_dim
    self.e_params = e_params
    self.d_params = d_params
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.lr = lr

  def fit(self, data, categorical_columns = []):
    continous, categorical = self.__split(data, categorical_columns)
    self.categorical_columns = categorical_columns
    self.continous_columns = list(continous.columns)

    self.continous_scaler.fit(continous)
    self.categorical_scaler.fit(categorical)
    continous, categorical = self.__transform(continous, categorical)
    d1 = pd.DataFrame(continous, columns=self.continous_columns)
    data = d1.join(
        pd.DataFrame(categorical, columns=self.categorical_scaler.get_feature_names_out())
    )

    self.data_dim = data.shape[1]
    self.encoder = Encoder(self.data_dim, self.latent_dim, self.e_params)
    self.decoder = Decoder(self.data_dim, self.latent_dim, self.d_params)

    self.__train(data)

  def __transform(self, continous, categorical):
    return self.continous_scaler.transform(continous), self.categorical_scaler.transform(categorical).toarray().astype(np.float64)

  def __split(self, data, categorical_columns):
    categorical = data[categorical_columns]
    continous = data.drop(categorical_columns, axis=1)
    return continous, categorical

  def __train_loader(self, data):    
    d = torch.tensor(data.values)
    d = d.type(dtype=torch.float32)
    d_length = d.size(dim=0)
    d_labels = torch.zeros(d_length)
    train_loader = torch.utils.data.DataLoader(
        d, batch_size=self.batch_size, shuffle=True
    )
    return train_loader

  def __train(self, data):
    optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr = self.lr, weight_decay=1e-6)
    self.training = True
    data_loader = self.__train_loader(data)

    for epoch in range(self.num_epochs):
      for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        mu, logvar = self.encoder(batch)
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decoder(z)

        loss = self.loss_function(batch, reconstruction, mu, logvar)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 and i == 0:
          print(f"Epoch: {epoch}, loss: {loss}")

  def reparametrize(self, mu, logvar):
    if self.training:
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return eps * std + mu
    return mu

  def loss_function(self, data, reconstruction, mu, logvar):
    cross_entropy = F.mse_loss(input=reconstruction, target=data, reduction='sum')
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return cross_entropy - KLD

  def sample(self, quantity = 1000):
    self.training = False
    with torch.no_grad():
      latent_samples = torch.randn(quantity, self.latent_dim)
      generated_samples = self.decoder(latent_samples)
      df = pd.DataFrame(generated_samples, columns = list(self.continous_columns) + list(self.categorical_scaler.get_feature_names_out()))
      continous, categorical = self.__split(df, self.categorical_scaler.get_feature_names_out())
      
      if self.continous_columns:
        scaled_continous = self.continous_scaler.inverse_transform(continous)
        if self.categorical_columns:
          scaled_categorical = self.categorical_scaler.inverse_transform(categorical)
          return np.concatenate((scaled_continous, scaled_categorical), axis=1)
        return scaled_continous
      scaled_categorical = self.categorical_scaler.inverse_transform(categorical)
      return scaled_categorical
