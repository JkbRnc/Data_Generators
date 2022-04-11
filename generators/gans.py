import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import seaborn as sn

import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
  def __init__(self, latent_dim = 64, d_params=[256, 256], g_params=[256, 256], num_epochs = 300, batch_size = 512, loss_function = nn.BCELoss(), lr = 0.002):
      self.discriminator = None
      self.generator = None
      self.scaler = MinMaxScaler()
      self.d_params = d_params
      self.g_params = g_params
      self.latent_dim = latent_dim
      self.num_epochs = num_epochs
      self.batch_size = batch_size
      self.loss_function = loss_function
      self.lr = lr

  def fit(self, data):
      self.scaler.fit(data)
      self.data_dim = data.shape[1]
      self.discriminator = Discriminator(self.data_dim, self.d_params)
      self.generator = Generator(self.latent_dim, self.data_dim, self.g_params)

  def transform(self, data):
      return self.scaler.transform(data)

  def train_loader(self, data):
      d = self.transform(data)
      d = torch.from_numpy(d)
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

  def train(self, data, k = 5, l = 1):
    optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
    optimizer_generator     = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-6)
    data_loader = self.train_loader(data)

    for epoch in range(self.num_epochs):
      for n, (real_samples, _) in enumerate(data_loader):
        batch_size = real_samples.size(dim=0)
        real_samples_labels = torch.ones((batch_size, 1))        

        # discriminator training cycle
        for i in range(k):
          latent_space_samples = torch.randn((batch_size, self.latent_dim), dtype=torch.float)

          generated_samples = self.generator(latent_space_samples)
          generated_samples_labels = torch.zeros((batch_size, 1))

          all_samples = torch.cat(
              (real_samples, generated_samples)
          )
          all_samples_labels = torch.cat(
              (real_samples_labels, generated_samples_labels)
          )

          self.discriminator.zero_grad()
          output_discriminator = self.discriminator(all_samples)
          loss_discriminator = self.loss_function(
              output_discriminator, all_samples_labels
          )
          loss_discriminator.backward()
          optimizer_discriminator.step()

        # data for generator training
        for i in range(l):
          latent_space_samples = torch.randn((batch_size, self.latent_dim))

          # generator training
          self.generator.zero_grad()
          generated_samples = self.generator(latent_space_samples)
          output_discriminator_generated = self.discriminator(generated_samples)
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
    generated_samples = self.generator(latent_samples)
    return self.scaler.inverse_transform(generated_samples.detach())
