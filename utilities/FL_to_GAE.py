from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable

import networkx as nx
from scipy.sparse import csr_matrix
from collections import defaultdict

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from .utils import eval_gae, make_sparse
from .models import GAE
from .preprocessing import mask_test_edges, preprocess_graph




def fl_to_gae(parameters,  FL_adj):
    A_new = nx.from_numpy_matrix(FL_adj)
    graph_data = list(A_new.edges())
    number_of_train = 3
    total_clients, num_features = parameters.shape

    post_parameters = np.where(parameters < 0.5, 0, 1)
    train_parameters = post_parameters[0:number_of_train, :]
    test_parameters = post_parameters[number_of_train: total_clients, :]
    allx = csr_matrix(train_parameters)
    tx = csr_matrix(test_parameters)
    test_idx_reorder = list(range(number_of_train, total_clients))
    test_idx_range = np.sort(test_idx_reorder)
    graph = defaultdict(list)

    for (key, value) in graph_data:
        graph[key].append(value)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    N, D = features.shape
    # Store original adjacency matrix (without diagonal entries)
    adj_orig = adj
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    # print(adj_train)

    # Some preprocessing
    adj_train_norm = preprocess_graph(adj_train)
    adj_train_norm = Variable(make_sparse(adj_train_norm))
    adj_train_labels = Variable(torch.FloatTensor(adj_train + sp.eye(adj_train.shape[0]).todense()))
    features = Variable(make_sparse(features))

    """:hyper-parameters"""
    # seed = 2
    dropout = 0.0
    # args.num_epochs  = 50
    num_epochs = 10
    # args.dataset_str = 'cora'
    test_freq = 10
    lr = 0.01


    #n_edges = adj_train_labels.sum()
    data = {
        'adj_norm': adj_train_norm,
        'adj_labels': adj_train_labels,
        'features': features,
    }

    gae = GAE(data,
              n_hidden=32,
              n_latent=16,
              dropout=dropout)

    optimizer = Adam({"lr": lr, "betas": (0.95, 0.999)})

    svi = SVI(gae.model, gae.guide, optimizer, loss=Trace_ELBO())

    # Results
    results = defaultdict(list)

    # Full batch training loop
    for epoch in range(num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        #epoch_loss = -epoch_loss
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step()

        # report training diagnostics
        normalized_loss = epoch_loss / (2 * N * N)
        results['train_elbo'].append(normalized_loss)

        # Training loss
        emb = gae.get_embeddings()
        accuracy, roc_curr, ap_curr = eval_gae(val_edges, val_edges_false, emb, adj_orig)


        # get adjancy matrix
        results['accuracy_train'].append(accuracy)
        results['roc_train'].append(roc_curr)
        results['ap_train'].append(ap_curr)
    z_adj = gae.model()  # the graph output from decoder, which is  torch size 2-D
    print("Optimization Finished!")

    L_G = nx.from_numpy_matrix(FL_adj)
    L = nx.laplacian_matrix(L_G)
    u, s, vh = np.linalg.svd(L.toarray(), full_matrices=True)


    z_numpy = z_adj.detach().numpy()
    z_G = nx.from_numpy_matrix(z_numpy)
    #Li_new = np.diag(z_numpy.diagonal()) - z_numpy
    Li =nx.laplacian_matrix(z_G)
    u_new, s_new, vh_new = np.linalg.svd(Li.toarray(), full_matrices=True)

    #np.random.seed(42)
    new_features =np.dot(u_new, np.dot(u.T, parameters))
    """mnist dataset"""
    w_attack = np.sum(new_features, axis=0) / new_features.shape[0] * np.random.uniform(-1.0, 1.0, new_features.shape[1]) # new_features.shape[0] is column

    """cifar10 dataset"""
    #w_attack = np.sum(new_features, axis=0) / new_features.shape[0] * np.random.normal(0.0, 1.0, new_features.shape[1]) # new_features.shape[0] is column


    # w_attack = np.sum(new_features, axis=0) / 100 # new_features.shape[0] is column
    dist = abs(np.linalg.norm(w_attack - new_features[[0], :]) - np.linalg.norm(w_attack - new_features[[1], :]))
    return w_attack,   z_numpy, normalized_loss, dist


