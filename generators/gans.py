import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import seaborn as sn

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Discriminator(nn.Module):
  def __init__(self, input_dim, hidden_layers_params):
    super(Discriminator, self).__init__()
    layers = []
    input = input_dim
    for layer_param in hidden_layers_params:
      layers += [nn.Linear(input, layer_param), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
      input = layer_param
    layers += [nn.Linear(input, 1)]
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    output = self.model(x)
    return torch.sigmoid(output)

class Generator(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_layers_params):
    super(Generator, self).__init__()
    layers = []
    input = input_dim
    for layer_param in hidden_layers_params:
      layers += [nn.Linear(input, layer_param), nn.ReLU()]
      input = layer_param
    layers += [nn.Linear(input, output_dim)]
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    output = self.model(x)
    return output

class GAN():
  def __init__(self, latent_dim = 64, d_params=[256, 256], g_params=[256, 256], num_epochs = 300, batch_size = 512, loss_function = nn.BCELoss(), lr = 0.002, k=5):
      self.__discriminator = None
      self.__generator = None
      self.__continous_scaler = StandardScaler()
      self.__categorical_scaler = OneHotEncoder()
      self.d_params = d_params
      self.g_params = g_params
      self.latent_dim = latent_dim
      self.num_epochs = num_epochs
      self.batch_size = batch_size
      self.loss_function = loss_function
      self.lr = lr
      self.k = k

  def fit(self, data, categorical_columns = []):
      continous, categorical = self.__split(data, categorical_columns)
      self.__categorical_columns = categorical_columns
      self.__continous_columns = list(continous.columns)

      self.columns = self.__continous_columns + self.__categorical_columns

      self.__continous_scaler.fit(continous)
      self.__categorical_scaler.fit(categorical)
      continous, categorical = self.__transform(continous, categorical)
      data = pd.DataFrame(continous, columns=self.__continous_columns).join(
          pd.DataFrame(categorical, columns=self.__categorical_scaler.get_feature_names_out())
      )

      self.data_dim = data.shape[1]
      self.__discriminator = Discriminator(self.data_dim, self.d_params)
      self.__generator = Generator(self.latent_dim, self.data_dim, self.g_params)

      self.__train(data, k=self.k)

  def __transform(self, continous, categorical):
      return self.__continous_scaler.transform(continous), self.__categorical_scaler.transform(categorical).toarray().astype(np.float64)

  def __split(self, data, categorical_columns):
      categorical = data[categorical_columns]
      continous = data.drop(categorical_columns, axis=1)
      return continous, categorical

  def __train_loader(self, data):
      d = torch.tensor(data.values)
      d = d.type(dtype=torch.float32)
      d_length = d.size(dim=0)
      d_labels = torch.zeros(d_length)
      train_set = [
                   (d[i], d_labels[i]) for i in range(d_length)
      ]
      train_loader = torch.utils.data.DataLoader(
          train_set, batch_size=self.batch_size, shuffle=True
      )
      return train_loader

  def __train(self, data, k = 5, l = 1):
    optimizer_discriminator = torch.optim.Adam(self.__discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
    optimizer_generator     = torch.optim.Adam(self.__generator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
    data_loader = self.__train_loader(data)

    for epoch in range(self.num_epochs):
      for n, (real_samples, _) in enumerate(data_loader):
        batch_size = real_samples.size(dim=0)
        real_samples_labels = torch.ones((batch_size, 1))        

        for i in range(k):
          latent_space_samples = torch.randn((batch_size, self.latent_dim), dtype=torch.float)

          generated_samples = self.__generator(latent_space_samples)
          generated_samples_labels = torch.zeros((batch_size, 1))

          all_samples = torch.cat(
              (real_samples, generated_samples)
          )
          all_samples_labels = torch.cat(
              (real_samples_labels, generated_samples_labels)
          )

          # discriminator training
          self.__discriminator.zero_grad()
          output_discriminator = self.__discriminator(all_samples)
          loss_discriminator = self.loss_function(
              output_discriminator, all_samples_labels
          )
          loss_discriminator.backward()
          optimizer_discriminator.step()

        # data for generator training
        for i in range(l):
          latent_space_samples = torch.randn((batch_size, self.latent_dim))

          # generator training
          self.__generator.zero_grad()
          generated_samples = self.__generator(latent_space_samples)
          output_discriminator_generated = self.__discriminator(generated_samples)
          loss_generator = self.loss_function(
              output_discriminator_generated, real_samples_labels
          )
          loss_generator.backward()
          optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == 0:
          print(f"Epoch: {epoch} Loss Discriminator: {loss_discriminator}")
          print(f"Epoch: {epoch} Loss Generator: {loss_generator}")

  def sample(self, quantity = 1000):
    latent_samples = torch.randn(quantity, self.latent_dim)
    generated_samples = self.__generator(latent_samples)
    df = pd.DataFrame(generated_samples, columns = list(self.__continous_columns) + list(self.__categorical_scaler.get_feature_names_out()))
    continous, categorical = self.__split(df, self.__categorical_scaler.get_feature_names_out())
    if self.__continous_columns:
      scaled_continous = self.__continous_scaler.inverse_transform(continous)
      if self.__categorical_columns:
        scaled_categorical = self.__categorical_scaler.inverse_transform(categorical)
        return np.concatenate((scaled_continous, scaled_categorical), axis=1)
      return scaled_continous
    scaled_categorical = self.__categorical_scaler.inverse_transform(categorical)
    return scaled_categorical