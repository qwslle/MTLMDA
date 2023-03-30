import csv
import sys
import numpy as np
import scipy.sparse as sp


csv_path = '/data/disease name.csv'
with open(csv_path,'r',encoding='utf8')as fp:
     dis_idx=[i[1] for i in csv.reader(fp)]
dis_idx==np.array(dis_idx)
dis_map_idx = {j: i for i , j in enumerate(dis_idx)}


csv1_path = '/data/miRNA number.csv'
with open(csv1_path,'r',encoding='utf8')as fp:
     mrna_idx = [j[1] for j in csv.reader(fp)]
mrna_idx=np.array(mrna_idx)
mrna_map_idx={j: i for i , j in enumerate(mrna_idx)}


csv2_path = '/data/miRNA-disease association.csv'  #####读联合数据
with open(csv2_path,'r',encoding='utf8')as fp:
     mrna_dis_idx = [j for j in csv.reader(fp)]

adj_mrna_dis=[]
for i in mrna_dis_idx:           ####将mrna——disease的联合转为列表，并以编号节点的形式表示
     adj_mrna_dis.append([mrna_map_idx[i[0]],dis_map_idx[i[1]]])
"""构建一个疾病——RNA矩阵，疾病有384种，RNA有495种，对应的相互作用有5430，形成一个495*384的矩阵"""
adj_mrna_dis= np.array(adj_mrna_dis) ###将 疾病——mrna 的联合转换成一个矩阵
adj_mrna_dis = sp.coo_matrix((np.ones(adj_mrna_dis.shape[0]), (adj_mrna_dis[:, 0], adj_mrna_dis[:, 1])),shape=(495, 384),dtype=np.float32)

csv3_path = '/data/gens.csv'
with open(csv3_path,'r',encoding='utf8')as fp:
     gens_idx=np.array([i[1] for i in csv.reader(fp)])
gens_map_idx = {j: i for i , j in enumerate(gens_idx)}

adj_gens_dis=[]
csv4_path = '/data/diseaseid-disease-gene.csv'  #####读gen_disease联合数据
with open(csv4_path,'r',encoding='utf8')as fp:
     gens_dis_idx = [j for j in csv.reader(fp)]
for i in gens_dis_idx:
     adj_gens_dis.append([gens_map_idx[i[2]],dis_map_idx[i[1]]])
adj_gens_dis= np.array(adj_gens_dis)

adj_gens_dis = np.array(adj_gens_dis) ###构建 gens——disease 联合矩阵
"""构建一个基因——疾病矩阵，基因有4519个，疾病有384种，对应的相互作用有18818，形成一个384*4520的矩阵"""
adj_gens_dis = sp.coo_matrix((np.ones(adj_gens_dis.shape[0]), (adj_gens_dis[:, 1], adj_gens_dis[:, 0])),shape=(384, 4520),dtype=np.float32)


"""异构图构建完成，节点为疾病，基因，miRNA，关系有疾病——rna，疾病——gen"""

print("sssssss")
print("sssssss")
print("sssssss")
print("sssssss")
