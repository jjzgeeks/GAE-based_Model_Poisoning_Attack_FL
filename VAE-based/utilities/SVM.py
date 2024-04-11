import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle


class SVM:

    def __init__(self, X_train, y_train, X_test, y_test, n_iters, val=True, val_type='k_fold', k=5, opt='mini_batch_GD', batch_size = 30, learning_rate=0.001, lambda_param=0.01):

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.val = val
        self.val_type =val_type
        self.k =k

        self.opt = opt
        self.batch_size = batch_size

        self.w = np.array([])
        self.b = None

    def grad(self, x, y):
        if y * (np.dot(x, self.w) - self.b) >= 1:
            dw = self.lr * (2 * self.lambda_param * self.w)
            db = 0
        else:
            dw = self.lr * (2 * self.lambda_param * self.w - np.dot(x, y))
            db = self.lr * y
        return (dw, db)

    def loss(self):
        return np.mean([max(0, 1-x * y) for x, y in
                        zip(np.where(np.concatenate(self.y_train, axis=None) <= 0, -1, 1), self.predict())])

    def stochastic_GD(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        y_ = np.where(y_train <= 0, -1, 1)

        if self.w.size == 0 and self.b is None:
            self.w = np.zeros(n_features)
            self.b = 0

        w_best = np.zeros(n_features)
        b_best = 0

        acc_list = []
        for i in range(0, self.n_iters):
            for idx, x_i in enumerate(X_train):
                dw, db = self.grad(x_i, y_[idx])
                self.w -= dw
                self.b -= db

            if i % 10 == 0 and self.val:
                approx_w = np.dot(X_val, self.w) - self.b
                approx_w = np.sign(approx_w)
                res_w = np.where(approx_w < 0, 0, approx_w)

                approx_w_best = np.dot(X_val, w_best) - b_best
                approx_w_best = np.sign(approx_w_best)
                res_w_best = np.where(approx_w_best < 0, 0, approx_w_best)

                if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
                    w_best = copy.deepcopy(self.w)
                    b_best = copy.deepcopy(self.b)

    def batch_GD(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        y_ = np.where(y_train <= 0, -1, 1)

        if self.w.size == 0 and self.b is None:
            self.w = np.zeros(n_features)
            self.b = 0

        w_best = np.zeros(n_features)
        b_best = 0

        acc_list = []
        for i in range(0, self.n_iters):
            dw_sum = 0
            db_sum = 0
            for idx, x_i in enumerate(X_train):
                dw, db = self.grad(x_i, y_[idx])
                dw_sum += dw
                db_sum += db
            self.w -= (dw_sum / n_samples)
            self.b -= (db_sum / n_samples)

            if i % 10 == 0 and self.val:
                approx_w = np.dot(X_val, self.w) - self.b
                approx_w = np.sign(approx_w)
                res_w = np.where(approx_w < 0, 0, approx_w)

                approx_w_best = np.dot(X_val, w_best) - b_best
                approx_w_best = np.sign(approx_w_best)
                res_w_best = np.where(approx_w_best < 0, 0, approx_w_best)

                if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
                    w_best = copy.deepcopy(self.w)
                    b_best = copy.deepcopy(self.b)

    def mini_batch_GD(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        y_ = np.where(y_train <= 0, -1, 1)

        if self.w.size == 0 and self.b is None:
            self.w = np.zeros(n_features)
            self.b = 0

        w_best = np.zeros(n_features)
        b_best = 0

        acc_list = []

        # print(self.n_iters)

        for i in range(0, self.n_iters):
            # print(i)
            dw_sum = 0.0
            db_sum = 0.0
            s = 0
            for idx, x_i in enumerate(X_train):
                dw, db = self.grad(x_i, y_[idx])
                dw_sum += dw
                db_sum += db
                s += 1
                if s % self.batch_size == 0:
                    self.w -= (dw_sum / self.batch_size)
                    self.b -= (db_sum / self.batch_size)

            if i % 10 == 0 and self.val:
                approx_w = np.dot(X_val, self.w) - self.b
                approx_w = np.sign(approx_w)
                res_w = np.where(approx_w < 0, 0, approx_w)

                approx_w_best = np.dot(X_val, w_best) - b_best
                approx_w_best = np.sign(approx_w_best)
                res_w_best = np.where(approx_w_best < 0, 0, approx_w_best)

                if (accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w)):
                    w_best = copy.deepcopy(self.w)
                    b_best = copy.deepcopy(self.b)

    def cross_validation(self, val_split):

        X_train = np.concatenate((self.X_train[0], self.X_train[1]), axis=0)
        y_train = np.concatenate((self.y_train[0], self.y_train[1]), axis=0)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=1,
                                                          stratify=y_train)

        eval("self." + self.opt + "(X_train, y_train, X_val, y_val)")

    def k_fold_cross_validation(self):

        X = np.concatenate((self.X_train[0], self.X_train[1]), axis=0)
        y = np.concatenate((self.y_train[0], self.y_train[1]), axis=0)

        w_list = []
        b_list = []
        acc_list = []

        if self.w.size == 0 and self.b == None:
            w = np.zeros(self.X_train[0].shape[1])
            b = 0
        else:
            w = copy.deepcopy(self.w)
            b = self.b

        skf = StratifiedKFold(n_splits=self.k, shuffle=True)

        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            eval("self." + self.opt + "(X_train, y_train, X_val, y_val)")

            print(self.accuracy())
            w_list.append(self.w)
            b_list.append(self.b)

            test_w = np.dot(X_val, self.w) - self.b
            test_w = np.sign(test_w)
            res_val = np.where(test_w < 0, 0, test_w)

            acc_list.append(accuracy_score(y_val, res_val))

            self.w = copy.deepcopy(w)
            self.b = b

        self.w = copy.deepcopy(w_list[acc_list.index(max(acc_list))])
        self.b = b_list[acc_list.index(max(acc_list))]

    def fit(self):
        if self.val_type == 'k_fold' and self.val:
            self.k_fold_cross_validation()

        elif self.val_type == 'cross_val' and self.val:
            self.cross_validation(0.2)

        elif not self.val:
            X_train = np.concatenate((self.X_train[0], self.X_train[1]), axis=0)
            y_train = np.concatenate((self.y_train[0], self.y_train[1]), axis=0)
            X_train, y_train = shuffle(X_train, y_train)
            eval("self." + self.opt + "(X_train, y_train)")

    def predict(self):
        approx = np.dot(self.X_test, self.w) - self.b
        approx = np.sign(approx)
        return np.where(approx < 0, 0, approx)

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    # def loss(self):
    #   # print(np.concatenate(self.y_train,axis=None))
    #   # print(np.vstack(self.x_train).shape)#, np.where(np.sign(np.dot(self.x_train, w) - b))<0, 0, 1).shape)
    #   return np.mean([max(0, 1-x*y) for x, y in zip(np.where(np.concatenate(self.y_train,axis=None) <= 0, -1, 1), np.where(np.sign(np.dot(self.X_train, self.w) - self.b)<0, -1, 1 ))])
