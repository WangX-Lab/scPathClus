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


##----------------------------------------------------------

os.chdir('/working_path')
os.getcwd() # Check the working path

pathway_score = pd.read_csv("sc_PathwayScore_example.csv",index_col = 0) # Get the single-cell pathway score matrix
pathway_score.shape  # check the data shape of the pathway score matrix: row for cells and columns for pathways
sample_names = pathway_score.index
pathway_score = pathway_score.astype('float32') # Convert the data type to speed up calculations
pathway_score = minmax_scale(pathway_score) # Convert the pathway score to 0-1 before dimension reduction

ae_ = Autoencoder_model(in_dim=pathway_score.shape[1]) # Instantiate a model for class(Autoencoder_model)
ae_.aeBuild()

epochs = 50 # Set the number of epochs
batch_size = 64 # Set the number of batch_size
log_path = "/working_path/logs" # set the log path for TensorBoard
tensorBoard = TensorBoard(log_dir=log_path, histogram_freq=0)

autoencoder_fit = ae_.autoencoder.fit(
    pathway_score, pathway_score,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=0,
    callbacks=[tensorBoard]
)

## plot the loss curve
loss_values = autoencoder_fit.history["loss"] 
epochs  = list(range(1, 51))
epoch_loss = pd.DataFrame({'epoch': epochs, 'loss': loss_value})

plt.plot(epochs, loss_values, 'b', marker='o', linestyle='-')
plt.title('Loss curve')
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.grid(True)
plt.show()

## Use the model to generate low dimensional features
encoded_path = ae_.encoder.predict(pathway_score)
col_names = ["Feature" + str(ii + 1) for ii in range(64)] 
encoded_feature = pd.DataFrame(data=encoded_path, columns=col_names, index= sample_names)
encoded_feature.to_csv("encoded_feature.csv")


#-----------------------------------------------------------
## Use scanpy to perform pathway-based clustering

adata = sc.read_h5ad("sc_data_example.h5ad")
adata.obs.shape # (5000, 7)
adata.var.shape # (18008, 2)
adata.obs.head() 
adata.var.head()
adata.to_df().iloc[0:5,0:5]
adata.raw = adata

encoded_feature = encoded_feature.reindex(adata.obs.index)
encoded_feature = encoded_feature.values
adata.obsm["X_pathway"] = encoded_feature ## Add the matrix of encoded features into adata 

sc.pp.neighbors(adata, n_neighbors=15, use_rep = "X_pathway") 
sc.tl.tsne(adata,use_rep = "X_pathway")
sc.tl.leiden(adata,resolution=1) 

## plot the clustering result
sc.pl.tsne(adata, color=['leiden'],legend_loc='on data') 

## map the patient information to the tSNE plot
sc.pl.tsne(adata, color=['Patient'])

## map the ground truth cell type labels to the tSNE plot
sc.pl.tsne(adata, color=['Cell_type'])

## Annotate Cell types through cell marker genes
marker_genes_dict = {'ductal cell 1': ['AMBP', 'CFTR', 'MMP7'],
                     'ductal cell 2': ['KRT19', 'KRT7', 'TSPAN8', 'SLPI'],
                     'ancinar': ['PRSS1', 'CTRB1','CTRB2','REG1B'],
                     'endocrine cell': ['CHGB', 'CHGA','INS','IAPP'],
                     'stellate cell': ['RGS5', 'ACTA2','PDGFRB','ADIRF'],
                     'fibroblast': ['LUM','DCN','COL1A1'],
                     'endothelial cell': ['CDH5','PLVAP','VWF','CLDN5'],
                     'macrophage': ['AIF1', 'CD14'],
                     'T cell': ['CD3D', 'CD4'],
                     'B cell': ['MS4A1','CD79A','CD79B','CD52']}
sc.pl.dotplot(adata, marker_genes_dict, groupby='leiden',use_raw = True)

new_cluster_names = ['Stellate cell_1','Macrophage cell_1', 'T/B cell_1','Ductal cell type 2_1', 'Ductal cell type 2_2','Fibroblast cell_1','Endothelial cell_1','Ductal cell type 1_1','Fibroblast cell_2','Ductal cell type 2_3','T/B cell_2','Endothelial cell_2','Ductal cell type 2_4','Ductal cell type 2_5','T/B cell_3']

adata.rename_categories('leiden', new_cluster_names)
adata.obs['leiden_pathway'] = adata.obs['leiden'].str[:-2]
sc.pl.tsne(adata, color=['leiden_pathway']) 
