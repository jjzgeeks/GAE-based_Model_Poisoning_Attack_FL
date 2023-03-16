import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import defaultdict

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and
# https://github.com/tkipf/gae
# ------------------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae(edges_pos, edges_neg, emb, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    emb = emb.data.numpy()
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []

    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []

    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    accuracy = accuracy_score((preds_all > 0.5).astype(float), labels_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return accuracy, roc_score, ap_score


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    """# load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, tx, allx, graph = tuple(objects)
 
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended"""
     
    if dataset == 'flparameters':
       parameters = np.random.rand(10,6)
       post_parameters = np.where(parameters < 0.5, 0, 1)
       train_parameters = post_parameters[0:6,:]
       test_parameters = post_parameters[6:10,:]
       allx = csr_matrix(train_parameters)
       tx = csr_matrix(test_parameters)
       test_idx_reorder = list(range(5,10))
       test_idx_range = np.sort(test_idx_reorder)
       graph = defaultdict(list)
       #graph_data  = [(0, 1), (1,2), (1,9), (2,1), (2,3), (3,4), (3,5), (4,3), (4,6), (5,3),(6,4), (6,7), (7,6),(7,8),(8,7),(8,9),(9,1),(9,8)]
       graph_data  = [(0, 1), (0,4), (0,5), (1,0), (1,4), (1,2), (2,1), (2,3), (2,6), (3,2),(3,5), (3,6),(4,0), (4,1),(4,5),(4,7),(5,0),(5,4),(5,8),(5,6),(5,3), (6,2), (6,3), (6,5), (6,9), (7,4), (7,8), (8,7), (8,5),(8,9),(9,8), (9,6)]
       #graph_data = [(0,9),(0,2),(1,8), (2,0), (2,5), (2,3), (3,2), (4,5),(5,4), (6,7), (7,6), (7,9), (8,1), (8,9), (9,0), (9,8), (9,7)]
       for (key, value) in graph_data:
           graph[key].append(value)
       #print(graph)
       features = sp.vstack((allx, tx)).tolil()
       features[test_idx_reorder, :] = features[test_idx_range, :]
       adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
       #print(adj)
    return adj, features
    """features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print(adj)
    return adj, features"""

def plot_results(results, test_freq, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'])
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss (ELBO)')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results['accuracy_train'])
    ax.plot(x_axis_test, results['accuracy_test'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig(path)
