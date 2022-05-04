from random import sample
from Data_Generators.generators.gans import GAN
from Data_Generators.generators.vae import VAE 
from os import path
import torch
import pandas as pd
import sys

def train(model_type, data, categorical_columns):
  if model_type == 'vae':
    model = VAE()
  else:
    model = GAN()
  df = pd.read_csv(data, sep=',')
  model.fit(df, categorical_columns=categorical_columns)
  return model

def generate(data, PATH, sample_size=100, categorical_columns=[], model_type='vae'):
  if not path.exists(PATH):
    model = train(model_type=model_type, data=data, categorical_columns=categorical_columns)
    torch.save(model, PATH)
  else:
    model = torch.load(PATH)
  samples = model.sample(sample_size)
  df = pd.DataFrame(samples, columns=model.columns)
  return df

def set_args(sample_size=100, categorical_columns="", model_type='vae'):
    categorical_columns = list(categorical_columns.split(','))
    sample_size = int(sample_size)
    return sample_size, categorical_columns, model_type

def main():
    args = sys.argv
    if len(args) < 3:
        print('Insufficient amount of arguments.')
        return
    data_path=args[1]
    size = len(args)
    match size:
      case 4:
        sample_size, categorical_columns, model_type = set_args(args[3])
      case 5:
        sample_size, categorical_columns, model_type = set_args(args[3], args[4])
      case 6:
        sample_size, categorical_columns, model_type = set_args(args[3], args[4], args[5])
      case _:
        sample_size, categorical_columns, model_type = set_args()
    df = generate(data=data_path, PATH=args[2], sample_size=sample_size, categorical_columns=categorical_columns, model_type=model_type)
    df.to_csv('generated_samples.csv')
    
    

if __name__ == "__main__":
    main()