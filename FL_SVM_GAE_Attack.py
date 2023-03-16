import numpy as np
import random
from random import randrange
import copy
import time
from sklearn.metrics import accuracy_score
import networkx as nx
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils import shuffle
from utilities.SVM import SVM
from utilities.FL_to_GAE import fl_to_gae
from utilities.data_processing import load_mnist_return_required_digits, get_clients, get_total_from_clients



class Federated_SVM:
    def __init__(self, x_train ,y_train, n_clients, n_iters, val=True, val_type='k_fold', k=5, opt='mini_batch_GD',
                 batch_size=30, learning_rate=0.001, lambda_param=0.01):
        self.n_clients = n_clients
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.val = val
        self.val_type = val_type
        self.client_distribution = [] # [3372, 3928, 3721] data size of each client
        self.k = k
        self.opt = opt
        self.batch_size = batch_size
        self.X_test = None
        self.y_test = None
        self.x_train = x_train
        self.y_train = y_train
        self.Loss = []
        # self.Loss_clients = [[0], [0], [0]]
        array_loss_clients = np.ones((self.n_clients,1))*0
        self.Loss_clients = array_loss_clients.tolist()
        self.timefit = []

    def create_clients(self, X_train, y_train, X_test, y_test):
        self.clients = []
        for i in range(self.n_clients):
            self.client_distribution.append(X_train[i][0].shape[0] + X_train[i][1].shape[0])
            self.clients.append(SVM(X_train[i], y_train[i], X_test, y_test, self.n_iters, self.val, self.val_type, self.k, self.opt, self.batch_size,
                     self.learning_rate, self.lambda_param))
        self.X_test = copy.deepcopy(X_test)
        self.y_test = copy.deepcopy(y_test)

    def average_aggregator(self, parameter_list):
        w = np.zeros(parameter_list[0].shape[0])
       # b = 0
        for i in range(0, self.n_clients):
            w = np.add(w, parameter_list[i] * self.client_distribution[i] / sum(self.client_distribution))
         #   b = b + parameter_list[i + 1] * self.client_distribution[i // 2] / sum(self.client_distribution)
        return w

    def loss(self, w):
        # print(np.concatenate(self.y_train,axis=None))
        # print(np.vstack(self.x_train).shape)#, np.where(np.sign(np.dot(self.x_train, w) - b))<0, 0, 1).shape)
        return np.mean([max(0, 1 - x * y) for x, y in zip(np.where(np.concatenate(self.y_train, axis=None) <= 0, -1, 1),
                                                          np.where(np.sign(np.dot(np.vstack(self.x_train), w)) < 0,
                                                                   -1, 1))])

    def fit(self, g_iters, aggregator):
        w_best = np.zeros(self.X_test.shape[1])
        for i in range(0, g_iters):
            print('global round', i + 1)
            for j in range(0, self.n_clients):
                if i == 0:
                    self.clients[j].fit()
                else:
                    self.clients[j].w = copy.deepcopy(w_agg)
                    self.clients[j].fit()
                self.Loss_clients[j].append(self.clients[j].loss())
                print('client', j + 1, self.clients[j].accuracy())
            parameter_list = []
            for k in range(0, self.n_clients):
                parameter_list.append(self.clients[k].w)  # all weights of clients

            # w_attack = FL_GAE(parameter_list)  ## GAE picks upa  new local weight as global weight
            w_agg = aggregator(parameter_list)  # global model averaged


            local_weights_array = np.array(parameter_list)
            #print(local_weights_array.shape) # 3*784
            if i == 0:
                initialization_graph_matrix = np.random.rand(self.n_clients, self.n_clients)
                initialization_graph = nx.from_numpy_matrix(initialization_graph_matrix)
                graph_data = list(initialization_graph.edges())
                w_attack, new_adj_matrix = fl_to_gae(local_weights_array, graph_data)
            else:
                graph_data = new_adj_matrix
                w_attack, new_adj_matrix = fl_to_gae(local_weights_array, graph_data)

            # calculate the distance
            dist = np.linalg.norm (w_attack - w_agg)
            dist_threshold = 0.0082
            print(dist)
            while (dist > dist_threshold):
                index = 0
                pick_num = 2 # random disturbance
                while (index < pick_num):
                    t = randrange(len(w_attack))
                    w_attack[t] = np.random.random()
                    index += 1
                dist = np.linalg.norm(w_attack - w_agg)
            # assign w_attack to w_agg
            w_agg = w_attack
            # print("agg",self.accuracy(w_agg,b_agg),"best",self.accuracy(w_best,b_best))
            if self.accuracy(w_agg) > self.accuracy(w_best) or i == 0:
                w_best = copy.deepcopy(w_agg)
               # b_best = copy.deepcopy(b_agg)
            self.Loss.append(self.loss(w_best))
            self.timefit.append(time.time())
            print('global test acc', self.accuracy(w_agg))
        return self.Loss

    def predict(self, w):
        approx = np.dot(self.X_test, w)
        approx = np.sign(approx)
        return np.where(approx < 0, 0, 1)

    def accuracy(self, w):
        return accuracy_score(self.y_test, self.predict(w)) * 100




if __name__ == '__main__':

    """Loading the data of 0 and 6"""
    n_clients = 5 #number of clients
    #n_clusters = 4
    data = load_mnist_return_required_digits(0, 6) # load data, image of digit 0 and digit 6
    # x_0_c1, y_0_c1, x_0_c2, y_0_c2, x_0_c3, y_0_c3, x_0_c4, y_0_c4 = create_kmeans_clusters(data[0][0], data[0][1], n_clusters)
    # x_6_c1, y_6_c1, x_6_c2, y_6_c2, x_6_c3, y_6_c3, x_6_c4, y_6_c4  = create_kmeans_clusters(data[1][0], data[1][1],  n_clusters)

    """Creation of individual train sets for the clients, 
    global train set for the SVM model and a global test set 
    containing the data from all the clusters"""
    # n_clients = n_clusters # number of clients
    clients_X, clients_y, X_test, y_test = get_clients(data[0][0], data[1][0], n_clients)
    xtrain_gl, ytrain_gl = get_total_from_clients(clients_X, clients_y)


    """ Batch global / SGD+Batch"""
    n_iters = 3 # number of local iterations
    n_global_commu_round = 20 # number of global communicaton round
    f_svm = Federated_SVM(xtrain_gl, ytrain_gl, n_clients, n_iters, val=False,  opt='batch_GD')

    f_svm.create_clients(clients_X, clients_y, X_test, y_test)
    Loss = f_svm.fit(n_global_commu_round, f_svm.average_aggregator)

    # plot loss curve
    plt.figure()
    plt.plot(range(n_global_commu_round), Loss)
    plt.ylabel('train_loss')
    plt.savefig('./fed_{}.png'.format(n_global_commu_round))
    plt.show()

