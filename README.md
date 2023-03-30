# MTLMDA
We propose an effective Multi-Task Learning forPredicting potential miRNA-disease Associations(MTLMDA),
which is an end-to-end trainable graph neural network model using GCN-based autoencoder and linear decoder.
In the MTLMDA, two sub-networks are constructed by miRNA-disease and gene-disease and are bridged
by specially designed cross&compress units. The whole network uses the information learned in the gene-disease
sub-network to assist the sub-network of miRNA-disease. MTLMDA obtained higher performance than the state-of-the-art method.

# We use pycharm as IDE and python as emulation language.
# Tested environment (The main package versions used in the experiment are as follows)
● dgl  =0.6.1

● numpy=1.19.5

● pandas=1.1.5

● python=3.6.13

● pytorch=1.10.2

# Hyperparameter settings
'--directory', default=data, The file name of the dataset;

'--seed', type=int, default=512, Random seed;

'--epochs', type=int, default=70,Number of epochs to train;

'--embedding_size', type=int, default=1024, node embedding dimension;

'--lr', type=float, default=0.0001, Initial learning rate;

'--weight_decay', type=float, default=3e-4, Weight decay (L2 loss on parameters);

'--dropout', type=float, default=0.3,Dropout rate (1 - keep probability).

# File explanation
data：The data file contains the miRNA-disease combination in the HMDD V2.0 dataset and the disease-gene associations we mined from the miRNA-disease network.

data1：The data1 file contains the miRNA-disease combination in the HMDD V3.2 dataset and the disease-gene associations we mined from the miRNA-disease network.

# How to run?
Run main.py

# How to train on different datasets？
At the directory position of the main program, set it to the new data set file address. For different data sets, it is only necessary to replace the file address value at directory positio, corresponding to the similarity of the imported network. Then return to the main program and click run.
