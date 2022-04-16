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

# class VAE(nn.Module):
#   def __init__(self, input_dim, latent_dim, encoder_hidden_layers_params, decoder_hidden_params):
#     super().__init__()
#     encoder_hidden_layers = []
#     decoder_hidden_layers = []
#     encoder_input = input_dim
#     decoder_input = latent_dim
#     for layer in encoder_hidden_layers_params:
#       encoder_hidden_layers += [
#                                 nn.Linear(encoder_input, layer),
#                                 nn.LeakyReLU()
#                                 # nn.ReLU()
#       ]
#       encoder_input = layer

#     for layer in decoder_hidden_params:
#       decoder_hidden_layers += [
#                                 nn.Linear(decoder_input, layer),
#                                 nn.LeakyReLU()
#                                 # nn.ReLU()
#       ]
#       decoder_input = layer
#     self.encoder = nn.Sequential(*encoder_hidden_layers)
#     self.encoder2mean   = nn.Linear(encoder_input, latent_dim)
#     self.encoder2logvar = nn.Linear(decoder_input, latent_dim)

#     decoder_hidden_layers.append(nn.Linear(decoder_input, input_dim))
#     self.decoder = nn.Sequential(*decoder_hidden_layers)

#   def encode(self, input):
#     # hidden_rep = input
#     hidden_rep = self.encoder(input)
#     # for layer in self.encoder:
#       # hidden_rep = F.relu(layer(hidden_rep))
#       # hidden_rep = layer(hidden_rep)
#     # print(f"Hidden rep: {hidden_rep}")
#     mu, logvar = self.encoder2mean(hidden_rep), self.encoder2logvar(hidden_rep)
#     # print(f"Mean: {mu}")
#     return mu, logvar

#   def decode(self, latent):
#     hidden_rep = self.decoder(latent)
#     # hidden_rep = latent
#     # for layer in self.decoder:
#     #   # hidden_rep = F.relu(layer(hidden_rep))
#     #   hidden_rep = layer(hidden_rep)
#     return hidden_rep

#   def reparametrize(self, mu, logvar):
#     if self.training:
#       std = torch.exp(logvar * 0.5)
#       eps = torch.randn_like(std)
#       return eps * std + mu
#     else:
#       return mu

#   def forward (self, x):
#     mu, logvar = self.encode(x)
#     z = self.reparametrize(mu, logvar)
#     return self.decode(z), mu, logvar

# class VAE(nn.Module):
#   def __init__(self, input_dim, latent_dim):
#     super().__init__()
#     hidden_dim = 64
#     # encoder layers
#     self.encoder_input2hidden  = nn.Linear(input_dim, hidden_dim)
#     self.encoder_hidden2mean   = nn.Linear(hidden_dim, latent_dim)
#     self.encoder_hidden2logvar = nn.Linear(hidden_dim, latent_dim)
#     # decoder layers
#     self.decoder_latent2hidden = nn.Linear(latent_dim, hidden_dim)
#     self.decoder_hidden2output = nn.Linear(hidden_dim, input_dim)

#   def encode(self, input):
#     # print(f"input: {input}")
#     hidden_rep = F.relu(self.encoder_input2hidden(input))
#     mu, logvar = self.encoder_hidden2mean(hidden_rep), self.encoder_hidden2logvar(hidden_rep)
#     # print(f"MU: {mu}")
#     # print(f"logvar: {logvar}")
#     return mu, logvar

#   def decode(self, latent):
#     hidden_rep = F.relu(self.decoder_latent2hidden(latent))
#     return self.decoder_hidden2output(hidden_rep)

#   def reparametrize(self, mu, logvar):
#     if self.training:
#       std = logvar.mul(0.5).exp_()
#       eps = std.data.new(std.size()).normal_()
#       return eps.mul(std).add_(mu)
#     else:
#       return mu

#   def forward (self, x):
#     mu, logvar = self.encode(x)
#     z = self.reparametrize(mu, logvar)
#     return self.decode(z), mu, logvar

# def vae_loss(data, reconstruction, mu, logvar):
#   # print(f"logvar: {logvar}")
#     MSE = F.mse_loss(input=reconstruction, target=data, reduction = 'sum')
#     KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return MSE - KLD
#   # loss = nn.BCELoss(reduction='sum')
#   # MSE = loss(reconstruction, data)
#   # KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
#   # return MSE + KLD

# def train_model(model, num_epochs, lr, data_loader):
#   optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=1e-6)
#   model.train()
#   for epoch in range(num_epochs):
#     train_loss = 0
#     for i, data in enumerate(data_loader):
#       optimizer.zero_grad()
#       reconstruction, mu, logvar = model(data)
#       loss = vae_loss(data, reconstruction, mu, logvar)
#       loss.backward()
#       train_loss += loss.item()
#       optimizer.step()
      

#     if epoch % 10 == 0:
#       print(f"Epoch: {epoch}, Average loss: {train_loss/len(train_loader_foe.dataset)}")
#     # print(f"Debug: Train loss: {train_loss}, Lenght: {len(train_loader_foe.dataset)}")
#     # print(f"MU: {mu}, logvar: {logvar}")
#     # print(f"Data {data}")

"""# Iris"""

iris = pd.read_csv('https://raw.githubusercontent.com/yangzhangalmo/pytorch-iris/master/dataset/iris.csv', sep=',')

iris.head()

# mapping = {
#     'Iris-setosa': 0,
#     'Iris-versicolor': 1,
#     'Iris-virginica': 2    
# }
# iris['species'] = iris['species'].apply(lambda x : mapping[x])
# # iris = iris.drop(['species'], axis = 1)

iris2 = iris.drop(['species'], axis=1)

iris_vae = VAE(latent_dim=10, e_params=[64], d_params=[64], lr =0.001)

iris_vae.fit(iris, categorical_columns=['species'])

samples = iris_vae.sample()

s = samples
df = pd.DataFrame(samples, columns=list(iris_vae.continous_columns) + list(iris_vae.categorical_columns))
df2 = df.drop(['species'], axis=1).astype(float)
corrMatrix_iris = df2.corr()
sn.heatmap(corrMatrix_iris, annot=True)
plt.show()

scaler.fit(iris)
iris_scaled = scaler.transform(iris)

# train_data = torch.tensor(iris_scaled.values)
train_data = torch.from_numpy(iris_scaled)
train_data_length = train_data.size(dim=0)
train_data_dim    = train_data.size(dim=1)
train_labels = torch.zeros(train_data_length)
train_data = train_data.type(dtype=torch.float32)
train_set = [
             (train_data[i], train_labels[i]) for i in range(train_data_length)
]

train_data_dim

latent_size = 3

iris_vae = VAE(train_data_dim, latent_size, [64], [64])

lr = 0.001
num_epochs = 300
batch_size = 30

train_loader_iris = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True
)

train_model(iris_vae, num_epochs, lr, train_loader_iris)

iris_vae.eval()
with torch.no_grad():
  latent_samples = torch.randn(1000, latent_size)
  samples = iris_vae.decode(latent_samples)
  samples = scaler.inverse_transform(samples)
  # print(samples)

# nmp = samples.detach().numpy()
df = pd.DataFrame(samples)
corrMatrix_iris = df.corr()
sn.heatmap(corrMatrix_iris, annot=True)
plt.show()

sn.heatmap(iris.corr(), annot=True)
plt.show()

"""# FOE"""

foe = pd.read_csv('drive/MyDrive/Datasets/FOE.csv', sep=',')

foe.head()

nums = list(range(0, len(foe['Date'])))
begin = foe['Date'][0]

foe['Date'] = nums

# foe = foe.drop(['Date'], axis=1)

foe.head()

vae = VAE(lr=0.001)

vae.fit(foe)

samples_foe = vae.sample()
df = pd.DataFrame(samples_foe, columns=list(vae.continous_columns) + list(vae.categorical_columns))
corrMatrix_foe = df.corr()
sn.heatmap(corrMatrix_foe, annot=True)
plt.show()

corrMatrix = foe.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

"""# Beans"""

beans = pd.read_csv('drive/MyDrive/Datasets/Dry_Bean_Dataset.csv', sep=';')

# beans = beans.drop(['Class'], axis=1)

beans.head()

vae_beans = VAE()

vae_beans.fit(beans, categorical_columns=['Class'])

vae_beans.train(beans)

vae_beans.sample(10)

samples = vae_beans.sample(10000)
df_beans = pd.DataFrame(samples, columns = list(vae_beans.continous_columns) + list(vae_beans.categorical_columns)).drop(['Class'], axis=1).astype(float)
corrMatrix_beans = df_beans.corr()
sn.heatmap(corrMatrix_beans, annot=True)
plt.show()

corr = beans.corr()
sn.heatmap(corr, annot=True)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

rfc = RandomForestClassifier()

df = pd.DataFrame(samples, columns = list(vae_beans.continous_columns) + list(vae_beans.categorical_columns))
train_data = df.drop(['Class'], axis=1).to_numpy()
train_labels = df['Class']
test_data = beans.drop(['Class'], axis=1).to_numpy()
test_labels = beans['Class'].to_numpy()

rfc.fit(train_data, train_labels)

rfc.score(test_data, test_labels)

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

knn.score(test_data, test_labels)

