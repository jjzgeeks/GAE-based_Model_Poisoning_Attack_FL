import numpy as np
import torch
import copy
import time
from sklearn.metrics import accuracy_score
#import networkx as nx
import matplotlib.pyplot as plt
from utilities.SVM import SVM
from utilities.FL_to_VAE import fl_to_vae
from utilities.data_processing import load_mnist_return_required_digits, get_clients, get_total_from_clients, load_cifar10_return_required_digits, create_kmeans_clusters, load_fashion_mnist_return_required_digits
from scipy.io import savemat

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)



class Federated_SVM:
    def __init__(self, x_train, y_train, n_clients, n_iters, val=True, val_type='k_fold', k=5, opt='mini_batch_GD',
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
        self.global_accuracy = []
        self.dist_set = []
        array_loss_clients = np.ones((self.n_clients,1))*0
        self.Loss_clients = array_loss_clients.tolist()
        self.local_clients_accuracy = array_loss_clients.tolist()
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
        for i in range(0, self.n_clients):
            w = np.add(w, parameter_list[i] * self.client_distribution[i] / sum(self.client_distribution))
        return w

    def loss(self, w):
        return np.mean([max(0, 1 - x * y) for x, y in zip(np.where(np.concatenate(self.y_train, axis=None) <= 0, -1, 1),
                                                          np.where(np.sign(np.dot(np.vstack(self.x_train), w)) < 0,
                                                                   -1, 1))])
    def fit(self, g_iters,  num_malicious):
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
                self.local_clients_accuracy[j].append(self.clients[j].accuracy())
                print('client', j + 1, self.clients[j].accuracy())
            """Partial local models are added noise """
            mean = np.random.random(self.n_clients - 3)
            #std = [0.1 for _ in range(self.n_clients)]
            std = np.random.randint(2, 5, self.n_clients - 3) / 5
            p1 = [self.clients[k].w * np.random.normal(mean[k], std[k], len(w_best))  for k in range(0, self.n_clients-3)]   # all weights of clients
            p2 = [self.clients[i].w for i in range(self.n_clients-3, self.n_clients)]
            parameter_list = p1 + p2

            #parameter_list =  [self.clients[k].w  for _ in range(0, self.n_clients)]

            local_weights_array = np.array(parameter_list) # benign clients; print(local_weights_array.shape) # n*784, n is the number of clients
            w_benign_global =  copy.deepcopy(np.sum(local_weights_array , axis=0) / local_weights_array.shape[0])

            num_select = 5 #5, 10, 15, 20
            random_pick_indices = np.random.choice(self.n_clients, size=num_select, replace=False)
            select_local_weights = copy.deepcopy(local_weights_array[random_pick_indices])

            #generates malicious clients
            w_attack_set = []
            for mali in range(num_malicious):
                w_attack  = fl_to_vae(select_local_weights)
                w_attack_set.append(w_attack)
            w_attack_arr = np.array(w_attack_set)
            w_all = np.row_stack((local_weights_array, w_attack_arr))
            w_agg = copy.deepcopy(np.sum(w_all, axis=0) / w_all.shape[0])
            distance = [np.linalg.norm(w_all[i, :] - w_benign_global) for i in range(self.n_clients + num_malicious)]
            self.dist_set.append(distance)


            #if self.accuracy(w_agg) > self.accuracy(w_best) or i == 0:
            if i == 0:
            #if i == 0:
                w_best = copy.deepcopy(w_agg)
            self.Loss.append(self.loss(w_best))
            self.timefit.append(time.time())
            print('global test acc', self.accuracy(w_agg))
            self.global_accuracy.append(self.accuracy(w_agg))
        return self.Loss, self.global_accuracy, self.local_clients_accuracy, self.dist_set



    def predict(self, w):
        approx = np.dot(self.X_test, w)
        approx = np.sign(approx)
        return np.where(approx< 0, 0, 1)

    def accuracy(self, w):
        return accuracy_score(self.y_test, self.predict(w))


if __name__ == '__main__':
    #dataset = ["mnist", "fashion_mnist", "cifar10"]
    dataset = ["mnist"]

    for x in dataset:
        # choose dataset
        if x == "mnist":
            """Loading the data"""
            data = load_mnist_return_required_digits(0, 6)  # load data, image of digit 0 and digit 6
        elif x == "fashion_mnist":
            data = load_fashion_mnist_return_required_digits(3, 8)
        else:
            """0:airplane; 1:automobile; 2:bird; 3:cat; 4:deer;
                5:dog; 6:frog; 7:horse; 8:ship; 9:truck"""
            data = load_cifar10_return_required_digits(1, 7)  # load data, image of label 1 and label 7
        """Creation of individual train sets for the clients, 
        global train set for the SVM model and a global test set 
        containing the data from all the clusters"""
        # n_clients = n_clusters # number of clients
        #num_clients_index = [5, 10, 15, 20, 25]
        num_clients_index = [5]
        #num_malicious_clients_index = [2, 4, 6, 8, 10]
        num_malicious_clients_index = [1]
        for n_clients in num_clients_index:
            clients_X, clients_y, X_test, y_test = get_clients(data[0][0], data[1][0], n_clients)
            xtrain_gl, ytrain_gl = get_total_from_clients(clients_X, clients_y)

            """ Batch global / SGD+Batch"""
            num_iters_index=[2,3,4,5,10]
            n_iters = num_iters_index[3] # number of local iterations

            num_global_commu_round_index = [100, 200, 300, 400, 500]
            n_global_commu_round =  num_global_commu_round_index[0] # number of global communicaton round


            for m_clients in num_malicious_clients_index:
                f_svm = Federated_SVM(xtrain_gl, ytrain_gl, n_clients, n_iters, val=False,  opt='batch_GD')
                f_svm.create_clients(clients_X, clients_y, X_test, y_test)
                clients = copy.deepcopy(f_svm.clients)
                Loss, global_accuracy,local_clients_accuracy, dist_set = f_svm.fit(n_global_commu_round, m_clients)

                # # plot global accuracy
                plt.figure()
                plt.plot(range(n_global_commu_round), global_accuracy)
                plt.xlabel('Communication rounds')
                plt.ylabel('Accuracy of global model')
                plt.savefig('./fed_glob_acc_{}_{}_{}_{}_{}.png'.format(x, n_clients,  n_global_commu_round, n_iters, m_clients))
                # #plt.savefig('./fed_glob_acc_{}_{}_{}_{}.eps'.format(x,  n_clients, n_global_commu_round, n_iters))
                # # plot local accuracy
                plt.figure()
                #color_list = ['green', 'red', 'yellow', 'blue', 'cyan']
                # # #color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
                #label_list = ['Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5']
                for i in range(n_clients):
                    #plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
                    plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1])
                # plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('Accuracy of clients')
                # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
                plt.savefig('./local_devices_accuracy_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round,n_iters, m_clients))
                # #plt.savefig('./FL_GAE_attack_results/local_devices_accuracy_{}_{}_{}_{}.eps'.format(x, n_clients, n_global_commu_round, n_iters))
                #
                #
                # # plot gae loss curve
                # plt.figure()
                # plt.plot(range(n_global_commu_round), gae_loss_list)
                # plt.xlabel('Communication rounds')
                # plt.ylabel('gae training loss')
                # plt.savefig('./FL_GAE_attack_results/gae_loss_{}_{}_{}.png'.format(x,  n_clients, n_global_commu_round, n_iters))
                #
                # # plot distance
                # plt.figure()
                # plt.plot(range(n_global_commu_round), distance)
                # plt.xlabel('Communication rounds')
                # plt.ylabel('distance')
                # plt.savefig('./FL_GAE_attack_results/distance_{}_{}_{}_{}.png'.format(x,  n_clients, n_global_commu_round, n_iters))

                plt.figure()
                #color_list = ['green', 'red', 'yellow', 'blue', 'cyan']
                # # #color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
                sty = ['-bs', '-g^']
                dist_set = np.array(dist_set)
                print(dist_set)
                for i in range(n_clients + m_clients):
                    if i < n_clients:
                       #plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
                       plt.plot(range(n_global_commu_round), dist_set[:, i])
                    else:
                       plt.plot(range(n_global_commu_round), dist_set[:, i], sty[i - n_clients], label=f'attacker {i - n_clients}')
                plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('Euclidean distance of clients')
                # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
                plt.savefig('./local_devices_Eucli_dist_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round,n_iters, m_clients))

                savemat("./FL_VAE_results_{}_{}_{}_{}_{}.mat".format(x, n_clients, n_global_commu_round, n_iters, m_clients), {"Global_model_loss": Loss, "Global_model_accuracy": global_accuracy,
                                 "Local_model_accuracy": local_clients_accuracy,"Euclidean distance set": dist_set})
                plt.show()
                plt.close()

