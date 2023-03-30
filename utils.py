import numpy as np
import pandas as pd
import dgl
import torch as th
import math

def gaussiansimilarity(interaction, m, n):

    gamad = n / (math.pow(np.linalg.norm(interaction), 2))
    C = interaction.T.conjugate()
    kd = np.zeros((n, n))

    D = np.matmul(C.T.conjugate(), C)
    for i in range(n):
        for j in range(i, n):
            kd[i][j] = np.exp(-gamad * (D[i][i] + D[j][j] - 2 * D[i][j]))

    kd = kd + kd.T.conjugate() - np.diag(np.diag(kd))

    # biomarker Gaussian
    gammam = m / (math.pow(np.linalg.norm(interaction), 2))
    km = np.zeros((m, m))
    E = np.matmul(C, C.T.conjugate())
    for i in range(m):
        for j in range(i, m):
            km[i][j] = np.exp(-gammam * (E[i][i] + E[j][j] - 2 * E[i][j]))
    km = km + km.T.conjugate() - np.diag(np.diag(km))
    return kd, km

def load_data():
######################HMDDV2.0
    D_GSM = np.loadtxt('data' + '/D_GSM.txt') ## 383*383,disease 高斯相似性 (disease——miRNA相互作用)
    M_GSM = np.loadtxt('data' + '/M_GSM.txt') ## 495*495,miRNA 高斯相似性(disease——miRNA相互作用)
    D_GSM2 = np.loadtxt('data' + '/Guss-dis.txt')  ##383*383,disease 高斯相似性 (disease——genes 相互作用)
    G_GSM=np.array(pd.read_csv('data' + '/G-g.csv', header=None)) ##4395*4395 genes高斯相似性 (disease——genes 相互作用)
######################HMDDV2.0

######################HMDDV3.0
    # D_GSM = np.array(pd.read_csv('data1' + '/D_GSM.csv',header=None))  ## 374*374,disease 高斯相似性 (disease——miRNA相互作用)
    # M_GSM = np.array(pd.read_csv('data1' + '/M_GSM.csv',header=None))  ## 788*788,miRNA 高斯相似性(disease——miRNA相互作用)
    # D_GSM2 = np.array(pd.read_csv('data1' + '/Guss-dis.csv',header=None))  ##374*374,disease 高斯相似性 (disease——genes 相互作用)
    # G_GSM = np.array(pd.read_csv('data1' + '/G-g.csv', header=None))  ##3384*3384 genes高斯相似性 (disease——genes 相互作用)
####提取miRNA-disease 和 gene-disease 相互作用矩阵
    

    return D_GSM,M_GSM,D_GSM2,G_GSM

def sample(directory,random_seed):
    DRassociations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    knownassociations=DRassociations.loc[DRassociations['label']==1]
    unknownassociations=DRassociations.loc[DRassociations['label']==0]
    DRnegativesample=unknownassociations.sample(n=knownassociations.shape[0],random_state=random_seed, axis=0)
    DRsample=knownassociations.append(DRnegativesample)
    DRsample.reset_index(drop=True,inplace=True)

    DGassociations=pd.read_csv(directory + '/all_genes_disease_pairs.csv', names=['genes', 'disease', 'label'])
    knownassociations2=DGassociations.loc[DGassociations['label']==1]
    unknownassociations2=DGassociations.loc[DGassociations['label']==0]
    DGnegativesample=unknownassociations2.sample(n=knownassociations2.shape[0],random_state=random_seed, axis=0)
    DGsample=knownassociations2.append(DGnegativesample)
    DGsample.reset_index(drop=True,inplace=True)
    return DRsample.values,DGsample.values

def build_graph(directory, random_seed):

    ID, IM, ID2, IG = load_data()
    train_matrix = np.array(pd.read_csv('data' + '/R-D.csv', header=None))
    # train_matrix=np.array(pd.read_csv('data1' + '/R-D.csv',header=None))
    # [n_m,n_d]=gaussiansimilarity(train_matrix, 374, 788)

    samples,samples2 =sample(directory, random_seed)
    # samples = np.array(pd.read_csv('data' + '/sample_sort.csv', header=None))

    print('Generating graph ...')

    g= dgl.DGLGraph()
    g2= dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    g2.add_nodes(IG.shape[0] + ID2.shape[1])

    node_type = th.zeros(g.number_of_nodes(), dtype=th.float32)
    node_type[:ID.shape[0]] = 1
    g.ndata['type'] = node_type

    node_type2 = th.ones(g2.number_of_nodes(), dtype=th.float32)
    node_type2[ID2.shape[0]:] = 2
    g2.ndata['type'] = node_type2

    print('Generating disease features ...' )
    d_data = th.zeros((g.number_of_nodes(), ID.shape[1]), dtype=th.float32)
    d_data[: ID.shape[0], :] = th.from_numpy(ID)
    g.ndata['d_features'] = d_data
    #
    print('Generating miRNA features ...' )
    m_data = th.zeros((g.number_of_nodes(), IM.shape[1]), dtype=th.float32)
    m_data[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = th.from_numpy(IM)
    g.ndata['m_features'] = m_data
    #
    print('Generating genes features ...')
    n_data = th.zeros((g2.number_of_nodes(), IG.shape[0]), dtype=th.float32)
    n_data[ID2.shape[0]: ID2.shape[0] + IG.shape[1], :] = th.from_numpy(IG)
    g2.ndata['g_features'] = n_data

    d_data2 = th.zeros((g2.number_of_nodes(), ID2.shape[1]), dtype=th.float32)
    d_data2[: ID.shape[0], :] = th.from_numpy(ID2)
    g2.ndata['d_features'] = d_data2

    print('Adding edges ...')
    disease_ids = list(range(1, ID.shape[0] + 1))
    mirna_ids = list(range(1, IM.shape[0] + 1))
    gene_ids=list(range(1,IG.shape[0]+1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}
    gene_ids_invmp={id_: i for i, id_ in enumerate(gene_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    sample_disease2_vertices=[disease_ids_invmap[i] for i in samples2[:,1]]
    sample_genes_vertices=[gene_ids_invmp[i]+ID.shape[0] for i in samples2[:,0]]

    g2.add_edges(sample_disease2_vertices,sample_genes_vertices,
                data={'d-g':th.ones(len(sample_genes_vertices)),
                      'rating': th.from_numpy(samples2[:, 2].astype('float32'))})
    g2.add_edges( sample_genes_vertices,sample_disease2_vertices,
                 data={'d-g': th.ones(len(sample_genes_vertices)),
                       'rating': th.from_numpy(samples2[:, 2].astype('float32'))})

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,  ##添加边，，构造无向图
                    data={'inv': th.ones(samples.shape[0], dtype=th.int32),
                          'rating': th.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                    data={'inv': th.ones(samples.shape[0], dtype=th.int32),
                          'rating': th.from_numpy(samples[:, 2].astype('float32'))})


    print('Successfully build graph !!')
    return g, g2, disease_ids_invmap, mirna_ids_invmap, gene_ids_invmp