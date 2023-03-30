import numpy as np
import pandas as pd

import torch
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn import EdgePredictor
from dgl.nn import ChebConv
from dgl.nn import TAGConv
from dgl.nn import GINConv



class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, diseases, mirnas):
        h = self.encoder(G)
        h_diseases = h[diseases]
        h_mirnas = h[mirnas]
        return self.decoder(h_diseases, h_mirnas)

class Model2(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, diseases, genes):
        h = self.encoder(G)
        h_diseases = h[diseases]
        h_genes = h[genes]
        return self.decoder(h_diseases, h_genes)

class Encoder(nn.Module):
    def __init__(self, G, embedding_size, dropout):
        super(Encoder, self).__init__()
        self.G=G

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.disease_emb = DiseaseEmbedding(G,embedding_size, dropout)
        self.mirna_emb = MirnaEmbedding(G,embedding_size, dropout)
        ######################HMDDV2.0
        self.miRNA_dis = torch.tensor(np.array(pd.read_csv('data' +'/R-D.csv', header=None)),dtype=torch.float)
        self.gene_dis =torch.tensor( np.array(pd.read_csv('data' +'/Gene-D.csv', header=None)),dtype=torch.float)
        self.CCU = CrossCompressUnit(self.miRNA_dis, self.gene_dis,embedding_size)
        ######################HMDDV2.0

        ######################HMDDV3.0####加载基于hmdd v3.2生成的gene——disease网络以及miRNA-disease网络
        # self.miRNA_dis = torch.tensor(np.array(pd.read_csv('data1' +'/R-D.csv', header=None)),dtype=torch.float)
        # self.gene_dis =torch.tensor( np.array(pd.read_csv('data1' +'/Gene-D.csv', header=None)),dtype=torch.float)
        # self.CCU = CrossCompressUnit(self.miRNA_dis, self.gene_dis,embedding_size)
        ######################HMDDV3.0
######()
        A1, A2, B1, B2 = self.CCU()
        G.apply_nodes(lambda nodes: {'h': self.disease_emb(nodes.data,B2)}, self.disease_nodes)
        G.apply_nodes(lambda nodes: {'h': self.mirna_emb(nodes.data,A1)}, self.mirna_nodes)
        self.node_feature = G.ndata['h']



        self.conv1 = ChebConv(embedding_size * 2, embedding_size, 2)  # 0.9382
        self.conv2 = ChebConv(embedding_size, 400, 2)
        self.conv3 = ChebConv(400, 64, 2)

    def forward(self, G):


        G.ndata['h']=self.node_feature

        h = self.conv1(G, G.ndata['h'])
        # h = F.relu(h)
        h = self.conv2(G, h)
        # h = F.relu(h)
        h = self.conv3(G, h)

        return h

class D_Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super(D_Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = (self.drop_layer(inputs)).to(torch.float32)
        output = self.fc(x)
        return output

class M_Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super(M_Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = (self.drop_layer(inputs)).to(torch.float32)
        output = self.fc(x)
        return output


class Encoder2(nn.Module):  ######222222222
    def __init__(self,G, embedding_size,dropout):
        super(Encoder2, self).__init__()
        self.G=G
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.genes_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 2)

        self.disease_emb = DiseaseEmbedding2(embedding_size, dropout)
        self.genes_emb = genesEmbedding(embedding_size, dropout)
        ######################HMDDV2.0
        self.miRNA_dis = torch.tensor(np.array(pd.read_csv('data' + '/R-D.csv', header=None)), dtype=torch.float)
        self.gene_dis = torch.tensor(np.array(pd.read_csv('data' + '/Gene-D.csv', header=None)), dtype=torch.float)
        ######################HMDDV2.0

        ######################HMDDV3.0
        # self.miRNA_dis = torch.tensor(np.array(pd.read_csv('data1' + '/R-D.csv', header=None)), dtype=torch.float)
        # self.gene_dis = torch.tensor(np.array(pd.read_csv('data1' + '/Gene-D.csv', header=None)), dtype=torch.float)
        ######################HMDDV3.0

        self.CCU = CrossCompressUnit(self.miRNA_dis, self.gene_dis, embedding_size)


        self.conv1 = SAGEConv(embedding_size,64, 'mean') ### 'mean'
        self.conv2 = SAGEConv(64, 64, 'mean')  ###'mean'
    def forward(self, G):

        A1, A2, B1, B2 = self.CCU()

        ####+++

        G.apply_nodes(lambda nodes:  {'h': self.disease_emb(nodes.data,A2)}, self.disease_nodes)
        G.apply_nodes(lambda nodes: {'h': self.genes_emb(nodes.data,B1)}, self.genes_nodes)

        h = self.conv1(G, G.ndata['h'])
        h = F.relu(h)
        h = self.conv2(G, h)
        return h

class MLP(nn.Module):
    def __init__(self, Martrix1, Martrix2):
        super(MLP, self).__init__()
        self.A = torch.cat([Martrix1, Martrix2], dim=1)
        self.W = nn.Parameter(torch.FloatTensor(self.A.shape[1], Martrix1.shape[1]))

    def forward(self):
        out = F.sigmoid(self.A)
        return out



class CrossCompressUnit(nn.Module):
    # def __init__(self,Martrix1,Martrix2):
    def __init__(self, Martrix1, Martrix2,embedding_size):
        super(CrossCompressUnit, self).__init__()
        self.Martrix1=Martrix1
        self.Martrix2 = Martrix2

        self.W1 = nn.Parameter(torch.FloatTensor(Martrix1.shape[0], embedding_size))
        self.W2 = nn.Parameter(torch.FloatTensor(Martrix1.shape[1], embedding_size))
        self.W3 = nn.Parameter(torch.FloatTensor(Martrix2.shape[0], embedding_size))
        self.W4 = nn.Parameter(torch.FloatTensor(Martrix2.shape[1],embedding_size))
        self.init_params()
    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
    def forward(self):

        output1=torch.mm(self.Martrix1,self.W2)
        output2=torch.mm(self.Martrix1.T,self.W1)
        output3 = torch.mm(self.Martrix2,self.W4)
        output4 = torch.mm(self.Martrix2.T,self.W3)
        return output1, output2,output3,output4

class genesEmbedding(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(genesEmbedding, self).__init__()
        seq = nn.Sequential(
            nn.Linear(4395, embedding_size),
            nn.Dropout(dropout)
        )

        self.proj_genes = seq
    def forward(self, ndata,crossfeatures):
        with torch.no_grad():
            # ndata['g_features']=0.9*ndata['g_features']+0.1*crossfeatures
            rep_genes = self.proj_genes(ndata['g_features'])   ###ndata['g_features']=4395*4395
            rep_genes=0.9*rep_genes+0.1*crossfeatures
        return rep_genes
class DiseaseEmbedding2(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(DiseaseEmbedding2, self).__init__()
        seq = nn.Sequential(
            nn.Linear(383, embedding_size),
            nn.Dropout(dropout)
        )
        self.proj_disease = seq

    def forward(self, ndata,crossfeatures):
        with torch.no_grad():
            rep_dis2 = self.proj_disease(ndata['d_features'])
            rep_dis2=0.9*rep_dis2+0.1*crossfeatures
        return rep_dis2

class DiseaseEmbedding(nn.Module):
    def __init__(self, G,embedding_size, dropout):
        super(DiseaseEmbedding, self).__init__()
        seq = nn.Sequential(
            nn.Linear(th.sum(G.ndata['type'] == 1), embedding_size),
            nn.Dropout(dropout)
        )
        self.proj_disease = seq

    def forward(self, ndata,crossfeatures):
        with torch.no_grad():
            rep_dis = self.proj_disease(ndata['d_features'])   ### ndata['d_features']=[383, 383]
            A=MLP(rep_dis,crossfeatures)
            rep_dis=A()

        return rep_dis

class MirnaEmbedding(nn.Module):
    def __init__(self, G,embedding_size, dropout):
        super(MirnaEmbedding, self).__init__()
        seq = nn.Sequential(
            nn.Linear(th.sum(G.ndata['type'] == 0), embedding_size),
            nn.Dropout(dropout)
        )

        self.proj_mirna = seq

    def forward(self, ndata,crossfeatures):
        with torch.no_grad():
            rep_mir = self.proj_mirna(ndata['m_features'])  ##ndata['m_features']=[495, 495]

            A1=MLP(rep_mir,crossfeatures)
            rep_mir=A1()
        return rep_mir

class Decoder(nn.Module):
    def __init__(self, feature_size):
        super(Decoder, self).__init__()

        self.activation=nn.Sigmoid()
        self.W = nn.Parameter(torch.FloatTensor(64,64))
        self.init_params()
    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self,h_diseases, h_mirnas):

        results_mask = self.activation(th.mm((h_diseases * h_mirnas), self.W).sum(1))

        return results_mask

class Decoder2(nn.Module):
    def __init__(self, feature_size):
        super(Decoder2, self).__init__()
        self.activation=nn.Sigmoid()
        self.W = nn.Parameter(torch.FloatTensor(64,64))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, h_diseases, h_genes):
        results_mask2 = self.activation((th.mm(h_diseases, self.W) * h_genes).sum(1))
        return results_mask2
