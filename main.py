import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from train import Train
import argparse
import torch
import warnings
import csv
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=512, help='Random seed.')
parser.add_argument('--epochs', type=int, default=70,
                    help='Number of epochs to train.')
parser.add_argument('--embedding_size', type=int, default=1024,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,  #0.0005  0.001--0.9389
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=3e-4, #1e-3  3e-4 0.9398
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')  #0.7 表示 70%的神经元随机被不激活  0.3时 AUC=0.9392

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    auc, acc, pre, recall, f1, fprs, tprs,auc2, acc2, pre2, recall2, f1_2, fprs2, tprs2,pres,recs,PR=Train(directory='data',
                                                  epochs=args.epochs,#100
                                                  embedding_size=args.embedding_size,
                                                  dropout=args.dropout,
                                                  lr=args.lr,
                                                  wd=args.weight_decay,
                                                  random_seed=args.seed
                                                   )
    print('model1-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc),np.std(auc)),
          'model1-Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc),np.std(acc)),
          'model1-Precision mean: %.4f , variance: %.4f\n' % (np.mean(pre),np.std(pre)),
          'model1-Recall mean: %.4f , variance: %.4f\n' % (np.mean(recall),np.std(recall)),
          'model1-F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1),np.std(f1)), '----------------------------------------------------------------------------------------------------\n' 
          'model2-AUC mean: %.4f \n' % (np.mean(auc2)),
          'model2-Accuracy mean: %.4f \n' % (np.mean(acc2)),'model2-Precision mean: %.4f \n' % (np.mean(pre2)),
          'model2-Recall mean: %.4f \n' % (np.mean(recall2)),'model2-F1-score mean: %.4f \n' % (np.mean(f1_2))
          )
    # plt.figure()
    plt.subplot(121)
    mean_fpr = np.linspace(0, 1, 3000)
    tpr=[]
    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, auc[i]))
    mean_tpr=np.mean(tpr,axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc)
    #######data restore
    # file_name = 'output2.csv'
    # f = open('output2.csv', 'a+', encoding='utf-8', newline="")
    # csv_writer = csv.writer(f)
    # for i in range(len(mean_fpr)):
    #     csv_writer.writerow([mean_fpr[i],mean_tpr[i]])
    # f.close()
    #######data restore
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f  $\pm$ %.4f)' % (mean_auc,auc_std))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # mean_tpr = np.mean(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr , 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')


    plt.subplot(122)
    mean_rec = []
    mean_pre = []

    for i in range(len(recs)):
        plt.plot(recs[i], pres[i], alpha=0.4, label='A-P ROC fold %d (AUC = %.4f)' % (i + 1, PR[i]))

    ##筛出最短的一个集合
    a=0
    for i in range(1,5):
        if len(recs[a])>len(recs[i]):
            a=i
    ##筛出最短的一个集合
    for i in range(len(recs[a])):
        mean_pre.append((pres[0][i] + pres[1][i] + pres[2][i] + pres[3][i] + pres[4][i]) / 5)
        mean_rec.append((recs[0][i] + recs[1][i] + recs[2][i] + recs[3][i] + recs[4][i]) / 5)

    mean_prauc = np.mean(PR, axis=0)
    plt.plot(mean_rec, mean_pre, color='b', alpha=0.8, label='Mean PRAUC (AUC = %.4f )' % (mean_prauc))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('recall Rate')
    plt.ylabel('precision Rate')
    plt.title(' P-R Curves')
    plt.legend(loc='lower right')

    plt.show()