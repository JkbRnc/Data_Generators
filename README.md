# Data_Generators

# About the project

Aim of this project is to create tabular generators for categorical and for numerical data. This is also practical part to my bachelor thesis.

# Generators

Currently there are inluded 2 modifications of generative models -- Generative Adversal Networks (Goodfellow et al., 2014) and Variational Autoencoder (Kingma et al., 2014)

# Usage

To generate data. First you need to install mandatory libraries (python setup.py install). Then just use generate_script.py to generate your data.
  
Mandatory arguments for generate_script.py:  
[DATA_PATH] - path to input data, can be absolute path or url  
[MODEL_PATH] - path to existing model or where to save it  
  
Optional arguments:  
[SAMPLE_SIZE] - size of generated data sample, default value set to 100 samples  
[CATEGORICAL_COLUMNS] - list of categorical columns in dataset, names separated only by comma  
[MODEL_TYPE] - name of desired model (vae or gans), by default set to vae  

# References

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; et al.: Generative adversal nets. In Advances in Neural Information Processing Systems, Curran Associates, Inc., 2014, s. 2672–2680. DOI: 
https://doi.org/10.48550/arXiv.1406.2661

Kingma, D. P.; Welling, M.: Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff AB, Canada, April 14-16, 2014, Conference Track Proceedings, 12 2014. DOI: https://doi.org/10.48550/arXiv.1312.6114
