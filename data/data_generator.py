# _*_ coding:utf-8 _*_
import gc
import pickle
from itertools import combinations
import math
import numpy as np
import torch
from joblib import Parallel, delayed


class Signal:
    def __init__(self, N, L, M, R, alpha, sigma2, c, num_train, num_val, num_test, n_jobs, data_path):
        self.N = N
        self.L = L
        self.M = M
        self.R = R
        self.alpha = alpha
        self.sigma2 = sigma2
        self.c = np.array(c.strip('[').strip(']').split(','), dtype=float)
        self.group_size = np.array(self.c).shape[0]
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.n_jobs = n_jobs
        self.data_path = data_path

    def mvb(self):
        tmp = np.zeros((2 ** self.group_size, self.group_size))
        for i in range(2 ** self.group_size, 2 ** (self.group_size + 1)):
            tmp[i - 2 ** self.group_size, :] = np.array(list(bin(i).replace("0b", "")[-self.group_size:]))

        index = [np.array(list(combinations(list(range(self.group_size)), i))) for i in range(1, self.group_size + 1)]
        prod = [tmp[:, index[i]].prod(axis=2).sum(axis=1) for i in range(len(index))]
        p = np.exp(self.c @ np.array(prod))

        return tmp, p / np.sum(p)

    def path_loss(self, R, alpha):
        # NP = R ** (-alpha) / (10 * self.M)  # noise power
        x_u_all = -R + 2 * R * np.random.uniform(0, 1, size=3 * self.N)
        y_u_all = -R + 2 * R * np.random.uniform(0, 1, size=3 * self.N)
        dist_u_all = np.sqrt(x_u_all ** 2 + y_u_all ** 2)
        dist_u_in = dist_u_all[(dist_u_all <= R) & (dist_u_all > 1)][0:self.N]
        g = dist_u_in ** (-alpha)

        return np.clip(g*1e6,1,100)

    def signal_generator(self, tmp, p, R, alpha):
        # S = np.random.normal(0, 1, size=(self.L, self.N)) + np.random.normal(0, 1, size=(self.L, self.N)) * 1j  #
        # Pilot matrix
        a = tmp[np.random.choice(list(range(2 ** self.group_size)), size=int(self.N / self.group_size), replace=True,
                                 p=p)].reshape(self.N)
        # p0 = p[2 ** (self.group_size - 1):].sum()
        # num_low = 0.9 * p0 * self.N
        # num_high = 1.1 * p0 * self.N
        # while a.sum() > num_high or a.sum() < num_low:
        #     a = tmp[
        #         np.random.choice(list(range(2 ** self.group_size)), size=int(self.N / self.group_size), replace=True,
        #                          p=p)].reshape(self.N)

        # A = np.diag(a)
        # User activity matrix
        g = self.path_loss(R, alpha)
        # G = np.diag(g)  # Large scale path loss
        H = torch.randn(self.N, self.M, dtype=torch.cfloat)  # Channel matrix
        Z = torch.randn(self.L, self.M, dtype=torch.cfloat) * math.sqrt(self.sigma2)  # AWGN
        # Y = S @ np.diag(a) @ np.diag(np.sqrt(g)) @ H + Z
        # Sigma_hat = Y @ Y.T.conjugate() / self.M
        return [torch.tensor(a, dtype=torch.float), torch.tensor(g, dtype=torch.float), H, Z]

    def train_val_test_generator(self):
        tmp, p = self.mvb()
        print(self.num_train, " train data generation starts!")
        data_train = Parallel(n_jobs=self.n_jobs)(
            delayed(self.signal_generator)(tmp, p, self.R, self.alpha) for _ in range(self.num_train))

        with open(self.data_path + "train" + ".pkl.bz2", 'wb') as f:
            pickle.dump(data_train, f)

        del data_train
        gc.collect()

        print(self.num_train, " train data generation ends!")

        print(self.num_val, " val data generation starts!")
        data_val = Parallel(n_jobs=self.n_jobs)(
            delayed(self.signal_generator)(tmp, p, self.R, self.alpha) for _ in range(self.num_val))

        with open(self.data_path + "val" + ".pkl.bz2", 'wb') as f:
            pickle.dump(data_val, f)

        del data_val
        gc.collect()
        print(self.num_val, " val data generation ends!")

        print(self.num_test, " test data generation starts!")
        data_test = Parallel(n_jobs=self.n_jobs)(
            delayed(self.signal_generator)(tmp, p, self.R, self.alpha) for _ in range(self.num_test))

        with open(self.data_path + "test" + ".pkl.bz2", 'wb') as f:
            pickle.dump(data_test, f)

        del data_test
        gc.collect()
        print(self.num_test, " test data generation ends!")

        print('Data generation is done!')
