import time
import random
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn import metrics
import csv

from sklearn.metrics import precision_recall_curve

import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
# from optimizer import Lookahead,RAdam

from model import  Model,Encoder,Decoder,Model2,Encoder2, Decoder2
from utils import build_graph, sample, load_data, gaussiansimilarity

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def Train(directory, epochs, embedding_size, dropout,  lr, wd, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)

    g, g2,disease_ids_invmap, mirna_ids_invmap,gene_ids_invmp= build_graph(directory, random_seed=random_seed)
    samples,samples2 = sample(directory, random_seed=random_seed)

    ID, IM, ID2, IG = load_data()

    print('## edges:', g.number_of_edges()+g2.number_of_edges())
    print('## disease nodes:', th.sum(g.ndata['type'] == 1))  ####383
    print('## mirna nodes:', th.sum(g.ndata['type'] == 0))   ####495
    print('##genes nodes:', th.sum(g2.ndata['type'] == 2))  ###4395

    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    samples_df2=pd.DataFrame(samples2, columns=['genes', 'disease', 'label'])
    sample_disease2_vertices=[disease_ids_invmap[i] for i in samples2[:,1]]
    sample_genes_vertices = [gene_ids_invmp[i] + ID.shape[0] for i in samples2[:, 0]]

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    train_index = []
    test_index = []
    train2_index=[]
    test2_index=[]

    for train_idx, test_idx in kf.split(samples[:, 2]):
        train_index.append(train_idx)
        test_index.append(test_idx)
    for train_idx, test_idx in kf.split(samples2[:, 2]):
        train2_index.append(train_idx)
        test2_index.append(test_idx)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []

    auc_result2 = []
    acc_result2 = []
    pre_result2 = []
    recall_result2 = []
    f1_result2 = []

    fprs = []
    tprs = []
    fprs2 = []
    tprs2 = []
    ###########plot P-R cruve #######
    pres = []
    recs = []
    PR = []
    ###########plot P-R cruve #######
    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)
############
        samples_df['train'] = 0
        samples_df['test'] = 0
        samples_df['train'].iloc[train_index[i]] = 1
        samples_df['test'].iloc[test_index[i]] = 1
        train_tensor = th.from_numpy(samples_df['train'].values.astype('int32'))
        test_tensor = th.from_numpy(samples_df['test'].values.astype('int32'))
        edge_data = {'train': train_tensor,
                     'test': test_tensor}


        g.edges[sample_disease_vertices, sample_mirna_vertices].data.update(edge_data)
        g.edges[sample_mirna_vertices, sample_disease_vertices].data.update(edge_data)
        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        # g_train = g.edge_subgraph(train_eid, preserve_nodes=True).to(device)
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        # rating_train = g_train.edata['rating'].to(device)
        rating_train = g_train.edata['rating']
        src_train, dst_train = g.find_edges(train_eid)
        # test_eid = g.filter_edges(lambda edges: edges.data['test']).to(device)
        test_eid = g.filter_edges(lambda edges: edges.data['test'])
        src_test, dst_test = g.find_edges(test_eid)
        # rating_test = g.edges[test_eid].data['rating'].to(device)
        rating_test = g.edges[test_eid].data['rating']

##########model2########
        samples_df2['train'] = 0
        samples_df2['test'] = 0
        samples_df2['train'].iloc[train2_index[i]] = 1
        samples_df2['test'].iloc[test2_index[i]] = 1
        train_tensor2 = th.from_numpy(samples_df2['train'].values.astype('int32'))
        test_tensor2 = th.from_numpy(samples_df2['test'].values.astype('int32'))
        edge_data = {'train2': train_tensor2,
                     'test2': test_tensor2}
        g2.edges[sample_disease2_vertices, sample_genes_vertices].data.update(edge_data)    ####识别出原图结构的训练集和测试集
        g2.edges[sample_genes_vertices, sample_disease2_vertices].data.update(edge_data)

        train_eid2 = g2.filter_edges(lambda edges: edges.data['train2'])
        g2_train = g2.edge_subgraph(train_eid2, preserve_nodes=True)
        rating_train2 = g2_train.edata['rating'].to(device)
        src_train2, dst_train2 = g2.find_edges(train_eid2)
        test_eid2 = g2.filter_edges(lambda edges: edges.data['test2'])
        src_test2, dst_test2 = g2.find_edges(test_eid2)
        rating_test2 = g2.edges[test_eid2].data['rating']

        print('## total Training edges:', len(train_eid)+len(train_eid2))
        print('## total Testing edges:', len(test_eid)+len(test_eid2))




        model=Model(Encoder(embedding_size=embedding_size, G=g_train, dropout=dropout),Decoder(feature_size=embedding_size))

        model2 = Model2(Encoder2(embedding_size=embedding_size, G=g2_train, dropout=dropout),
                        Decoder2(feature_size=embedding_size))

        cross_entropy = nn.BCELoss()
        # cross_entropy =nn.MSELoss()  ###用这个loss很低
        cross_entropy2=nn.MSELoss()



        trainer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # A=RAdam(model.parameters())
        # trainer=Lookahead(A,0.5,5)
        trainer2 = optim.Adam(model2.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs):
            start = time.time()
            for _ in range(10):

                    trainer.zero_grad()
                    trainer2.zero_grad()

                    score_train = model(g_train, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()
                    ####################      ####################
                    score_train2 = model2(g2_train, src_train2, dst_train2)
                    loss_train2 = cross_entropy2(score_train2, rating_train2).mean()  ##orsum()
                    loss_train1=loss_train2+loss_train

                    loss_train1.backward()
                    trainer.step()
                    trainer2.step()


            h_val2=model2.encoder(g2)
            score_val2 = model2.decoder(h_val2[src_test2], h_val2[dst_test2])
            loss_val2 = cross_entropy2(score_val2, rating_test2).mean()
            train_auc2 = metrics.roc_auc_score(np.squeeze(rating_train2.detach().numpy()),
                                              np.squeeze(score_train2.detach().numpy()))
            val_auc2 = metrics.roc_auc_score(np.squeeze(rating_test2.detach().numpy()),
                                            np.squeeze(score_val2.detach().numpy()))

            end = time.time()

            print('model2-Epoch:', epoch + 1, 'model2-Train Loss: %.4f' % loss_train2.item(),
                  'model2-Val Loss: %.4f' % loss_val2.item(),'model2-Train AUC: %.4f' % train_auc2, 'model2-Val AUC: %.4f' % val_auc2,
                  'TIME:%.2f' % (end - start))


###########################

            h_val = model.encoder(g)
            score_val =model.decoder(h_val[src_test], h_val[dst_test])

            loss_val = cross_entropy(score_val, rating_test).mean()

            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.detach().numpy()), np.squeeze(score_train.detach().numpy()))

            val_auc = metrics.roc_auc_score(np.squeeze(rating_test.detach().numpy()), np.squeeze(score_val.detach().numpy()))

            results_val = [0 if j < 0.5 else 1 for j in np.squeeze(score_val.detach().numpy())]

            accuracy_val = metrics.accuracy_score(rating_test.detach().numpy(), results_val)

            precision_val = metrics.precision_score(rating_test.detach().numpy(), results_val)

            recall_val = metrics.recall_score(rating_test.detach().numpy(), results_val)

            f1_val = metrics.f1_score(rating_test.detach().numpy(), results_val)

            end = time.time()

            print('model-Epoch:', epoch + 1, 'model-Train Loss: %.4f' % loss_train.item(),   ####loss_train.asscalar()
                  'model-Val Loss: %.4f' % loss_val.item(),  ###loss_val.asscalar()
                  'model-Acc: %.4f' % accuracy_val, 'model-Pre: %.4f' % precision_val, 'model-Recall: %.4f' % recall_val,
                  'model-F1: %.4f' % f1_val, 'model-Train AUC: %.4f' % train_auc, 'model-Val AUC: %.4f' % val_auc,
                  'Time: %.2f\n' % (end - start),'***************************************************************************************************************')

        model.eval()
        with torch.no_grad():
            h_test = model.encoder(g).to(device)
            score_test = model.decoder(h_test[src_test], h_test[dst_test]).to(device)
        fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(rating_test.detach().numpy()),
                                                 np.squeeze(score_test.detach().numpy()))
        ###########plot P-R cruve #######
        pre, rec, thresh = metrics.precision_recall_curve(np.squeeze(rating_test.detach().numpy()),
                                                          np.squeeze(score_test.detach().numpy()))
        AUPR = metrics.auc(rec, pre)
        PR.append(AUPR)

        ###########plot P-R cruve #######

        test_auc = metrics.auc(fpr, tpr)

        results_test = [0 if j < 0.5 else 1 for j in np.squeeze(score_test.detach().numpy())]
        accuracy_test = metrics.accuracy_score(rating_test.detach().numpy(), results_test)
        precision_test = metrics.precision_score(rating_test.detach().numpy(), results_test)
        recall_test = metrics.recall_score(rating_test.detach().numpy(), results_test)
        f1_test = metrics.f1_score(rating_test.detach().numpy(), results_test)

        print('Fold:', i + 1,
              'model-Test Acc: %.4f' % accuracy_test, 'model-Test Pre: %.4f' % precision_test,
              'model-Test Recall: %.4f' % recall_test, 'model-Test F1: %.4f' % f1_test, 'model-Test AUC: %.4f\n' % test_auc
              )

        auc_result.append(test_auc)
        acc_result.append(accuracy_test)
        pre_result.append(precision_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)

        fprs.append(fpr)
        tprs.append(tpr)
#########################

        f = open('MTLMDA.csv', 'a+', encoding='utf-8', newline="")
        csv_writer = csv.writer(f)
        for i in range(len(fpr)):
            csv_writer.writerow([fpr[i], tpr[i]])
        f.close()
####################
        pres.append(pre)
        recs.append(rec)

##################################
        model2.eval()
        with torch.no_grad():
            h_test2 = model2.encoder(g2)
            score_test2 = model2.decoder(h_test2[src_test2], h_test2[dst_test2])
        fpr2, tpr2, thresholds = metrics.roc_curve(np.squeeze(rating_test2.detach().numpy()),
                                                 np.squeeze(score_test2.detach().numpy()))

        test_auc2 = metrics.auc(fpr2, tpr2)

        results_test2 = [0 if j < 0.5 else 1 for j in np.squeeze(score_test2.detach().numpy())]
        accuracy_test2 = metrics.accuracy_score(rating_test2.detach().numpy(), results_test2)
        precision_test2 = metrics.precision_score(rating_test2.detach().numpy(), results_test2)
        recall_test2 = metrics.recall_score(rating_test2.detach().numpy(), results_test2)
        f1_test2 = metrics.f1_score(rating_test2.detach().numpy(), results_test2)

        print('Fold:', i + 1,
              'model2-Test Acc: %.4f' % accuracy_test2, 'model2-Test Pre: %.4f' % precision_test2,
              'model2-Test Recall: %.4f' % recall_test2, 'model2-Test F1: %.4f' % f1_test2, 'model2-Test AUC: %.4f' % test_auc2
              )

        auc_result2.append(test_auc2)
        acc_result2.append(accuracy_test2)
        pre_result2.append(precision_test2)
        recall_result2.append(recall_test2)
        f1_result2.append(f1_test2)

        fprs2.append(fpr2)
        tprs2.append(tpr2)
    print('####### Training Finished !#######')
    return auc_result, acc_result, pre_result, recall_result, f1_result, fprs, tprs,auc_result2, acc_result2, pre_result2, recall_result2, f1_result2, fprs2, tprs2,pres,recs,PR



