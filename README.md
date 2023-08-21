# scPathClus
A pathway-based clustering method for single cells.
![figure1](https://github.com/WangX-Lab/scPathClus/assets/54932820/3c7954ca-9ee6-4440-85b4-948eaa62dbd8)
# Description
To identify cell subpopulations with pathway heterogeneity, scPathClus requires a single-cell pathway enrichment score matrix. First, an autoencoder is used to obtain the low-dimensional pathway features. Then, the low-dimensional feature matrix is embedded into scanpy to perform pathway-based clustering. 
# Prerequisites
- Python 3.9
- pandas 1.5.3
- scanpy 1.7.2
- scikit-learn 1.2.1
- tensorflow 2.10.0
- TensorBoard 2.10.0
- keras  1.1.2
- numpy  1.23.5
- matplotlib 3.3.2
- os 1.3.1
# Examples
This sample data is a random subset (5,000 cells) from PDAC_CRA001160 dataset. 
## Load dependencies
```
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import tensorflow as tf
import scanpy as sc
import os
```
## Define a class for autoencoder
```
class Autoencoder_model:
    def __init__(self,in_dim):
        self.in_dim = in_dim
        self.autoencoder = None
        self.encoder = None
        
    def aeBuild( self ):
        in_dim = self.in_dim
        path_in = Input(shape = (self.in_dim,)) 
        
        # Encoder
        encoded_h1 = Dense(1024,activation = "relu", name = "enco1")(path_in)
        encoded_h2 = Dense(64,activation = "relu",name = "enco2")(encoded_h1)
        
        # decoder
        decoded_h1 = Dense(1024,activation = "relu",name = "deco1")(encoded_h2)
        decoded = Dense(self.in_dim,activation = "sigmoid",name = "deco2")(decoded_h1)
        
        # Connect the encoder and the decoder
        autoencoder = Model(inputs = path_in, outputs = decoded)
        
        # choose Adam as optimizer (learning_rate:0.00001)
        # loss function:MSE
        opt = Adam(learning_rate = 0.00001)
        autoencoder.compile(optimizer = opt, loss = "mean_squared_error")
        print(autoencoder.summary())
        encoder = Model(inputs = path_in, outputs = encoded_h2)
        self.autoencoder = autoencoder
        self.encoder = encoder
```
## Prepare data
```
os.chdir('/working_path')
os.getcwd() # Check the working path

pathway_score = pd.read_csv("sc_PathwayScore_example.csv",index_col = 0) 
pathway_score.shape  
sample_names = pathway_score.index
pathway_score = pathway_score.astype('float32') 
pathway_score = minmax_scale(pathway_score) 
```
- pathway_score：
  |                       | GOBP_MITOCHONDRIAL_GENOME_MAINTENANCE_UCell | GOBP_REPRODUCTION_UCell |
  | --------------------- | ------------- |--------------------- |
  | T12_ACACCCTTCCGTCATC  | 0.0911209677419355 | 0.412496516257465 |
  | T11_GATCAGTGTGCAACTT  | 0.0631693548387097  | 0.432461181154612 |
- Converting the data type from  float 64 to float 32 helps speed up the computation.
- Before performing the dimension reduction step, data should be normalized to 0-1. 
## 


















