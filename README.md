# scPathClus
A pathway-based clustering method for single cells.
![image](https://github.com/WangX-Lab/scPathClus/assets/54932820/25164681-65aa-460e-a11a-0454347e6767)

# Description
With the input of single-cell RNA sequencing data, scPathClus first uses [UCell](https://github.com/carmonalab/UCell) to transform the expression matrix into a single-cell pathway enrichment matrix. Then, an autoencoder is used to obtain the low-dimensional pathway latent feature matrix. Next, the latent feature matrix is embedded into scanpy to perform single-cell clustering. 
# Prerequisites
R:
- UCell 2.2.0
  
Python:
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
This sample data is a single-cell RNA expression matrix (18,008 genes x 5,000 cells) from PDAC_CRA001160 dataset. 
### Transform single-cell gene expression matrix into pathway enrichment matrix
+ Using R package "UCell" to perform pathway enrichment analysis.
+ input:<br>
(1) a single-cell gene expression matrix (18,008 genes x 5,000 cells) ``exp.mat`` <br>
(2) a pathway list (13259 pathways) ``pathway_list``
+ output: pathway enrichment matrix (5000 cells x 13259 pathways)
```R
## R
setwd("/working_path")
library(UCell)
u.scores <- ScoreSignatures_UCell(exp.mat, features = pathway_list,maxRank=2000)
write.csv(u.scores, "sc_PathwayScore_example.csv",row.names = T)
```
### Load dependencies
```python
## python
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
### Define a class for autoencoder
```python
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
### Load single-cell pathway enrichment matrix
```python
os.chdir('/working_path')
os.getcwd() # Check the working path

pathway_score = pd.read_csv("sc_PathwayScore_example.csv",index_col = 0)
```
- A brief example of pathway enrichment matrix：

  |                       | GOBP_MITOCHONDRIAL_GENOME_MAINTENANCE_UCell | GOBP_REPRODUCTION_UCell |
  | --------------------- | ------------- |--------------------- |
  | T12_ACACCCTTCCGTCATC  | 0.0911209677419355 | 0.412496516257465 |
  | T11_GATCAGTGTGCAACTT  | 0.0631693548387097  | 0.432461181154612 |
  
```python
pathway_score.shape  
sample_names = pathway_score.index
pathway_score = pathway_score.astype('float32') 
pathway_score = minmax_scale(pathway_score) 
```
- Converting the data type from  float 64 to float 32 helps speed up the computation.
- Before performing the dimension reduction step, data should be normalized to 0-1. 
### Instantiate an autoencoder model
```python
ae_ = Autoencoder_model(in_dim=pathway_score.shape[1]) 
ae_.aeBuild()
```
### Dimension reduction
```python
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
```
```python
### plot the loss curve
loss_values = autoencoder_fit.history["loss"] 
epochs  = list(range(1, 51))
epoch_loss = pd.DataFrame({'epoch': epoch, 'loss': ae_loss})

plt.plot(epochs, loss_values, 'b', marker='o', linestyle='-')
plt.title('Loss curve')
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.grid(True)
plt.show()
```
![image](https://github.com/WangX-Lab/scPathClus/assets/54932820/656073fe-68ee-4d90-a5ac-c38b3aba0b30)
- Or you can check the loss in real time by using tensorBoard in the callbacks function. 
```python
### Use the model to generate low dimensional latent features
encoded_path = ae_.encoder.predict(pathway_score)
col_names = ["Feature" + str(ii + 1) for ii in range(64)] 
encoded_feature = pd.DataFrame(data=encoded_path, columns=col_names, index= sample_names)
encoded_feature.to_csv("encoded_feature.csv")
```
- A brief example of latent feature matrix：
  |                       | Feature1 | Feature2 |Feature3 | Feature4 |Feature5 |
  | --------------------- | ------------- |----------------- |------------- |------------- |--------------- |
  | T12_ACACCCTTCCGTCATC  | 0 | 0 | 0 | 10.317576 | 0.6895857 |
  | T11_GATCAGTGTGCAACTT  | 0 | 0 | 0 | 0.09860481 | 0.90391874 |
  | T19_ACGATACGTTGTCGCG  | 0 | 0 | 0 | 8.977797 | 1.0264478 |

### Use scanpy to perform single-cell clustering
+ input:
(1) single-cell RNA sequencing data in the form of "anndata"
(2) a latent feature matrix 
```python
adata = sc.read_h5ad("sc_data_example.h5ad")
adata.obs.shape # (5000, 7)
adata.var.shape # (18008, 2)
adata.obs.head() 
adata.var.head()
adata.to_df().iloc[0:5,0:5]
adata.raw = adata

encoded_feature = encoded_feature.reindex(adata.obs.index)
encoded_feature = encoded_feature.values
adata.obsm["X_pathway"] = encoded_feature ## Add the matrix of latent features into adata 

sc.pp.neighbors(adata, n_neighbors=15, use_rep = "X_pathway") 
sc.tl.tsne(adata,use_rep = "X_pathway")
sc.tl.leiden(adata,resolution=1)
```
```python
## plot the clustering result
sc.pl.tsne(adata, color=['leiden'],legend_loc='on data')
```
![Figure_2](https://github.com/WangX-Lab/scPathClus/assets/54932820/7b9e9f98-dfde-4f28-880d-344122cab6b9)
```python
## map the patient information to the tSNE plot
sc.pl.tsne(adata, color=['Patient'])
```
![Figure_4](https://github.com/WangX-Lab/scPathClus/assets/54932820/2ac2bafd-404d-4ab1-8cab-d0ea000ab0b0)
```python
## map the ground truth cell type labels to the tSNE plot
sc.pl.tsne(adata, color=['Cell_type'])
```
![Figure_3](https://github.com/WangX-Lab/scPathClus/assets/54932820/d3f2592d-592b-49ed-9051-719fc677db32)
```python
## Annotate Cell types using cell marker genes
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
```
![Figure_5](https://github.com/WangX-Lab/scPathClus/assets/54932820/274d448b-b8aa-4d52-94e2-f9dd653a3780)

```python
new_cluster_names = ['Stellate cell_1','Macrophage cell_1', 'T/B cell_1','Ductal cell type 2_1', 'Ductal cell type 2_2','Fibroblast cell_1','Endothelial cell_1','Ductal cell type 1_1','Fibroblast cell_2','Ductal cell type 2_3','T/B cell_2','Endothelial cell_2','Ductal cell type 2_4','Ductal cell type 2_5','T/B cell_3']

adata.rename_categories('leiden', new_cluster_names)
adata.obs['leiden_pathway'] = adata.obs['leiden'].str[:-2]
sc.pl.tsne(adata, color=['leiden_pathway'])
```
![Figure_6](https://github.com/WangX-Lab/scPathClus/assets/54932820/70820e57-be07-4755-af34-2bace199ab65)

# Contact
E-mail any questions to Hongjing Ai hongjingai@stu.cpu.edu.cn or Xiaosheng Wang xiaosheng.wang@cpu.edu.cn















