import math
from itertools import permutations
import torch
import torch.nn as nn
import pickle
from utils import *

torch.manual_seed(2024)
class PSCANet(nn.Module):
    def __init__(self, N, L, M, sigma2, pilot, train_S, train_gamma, pathloss, num_layer=20):
        super(PSCANet, self).__init__()
        self.N = N
        self.L = L
        self.M = M
        self.num_layer = num_layer
        self.sigma2 = sigma2
        self.pathloss = pathloss
        self.register_buffer("I", torch.eye(self.L))

        tmp = torch.ones(num_layer) * 0.5
        for i in range(0, num_layer - 1):
            tmp[i + 1] = tmp[i] * (1 - 0.5 * tmp[i])

        if train_gamma == "Yes":
            self.gamma = nn.Parameter(tmp, requires_grad=True)
        else:
            self.register_buffer("gamma", 0.05 * torch.ones(self.num_layer))

        if pilot == "pretrain":
            with open("pilot/AMPNet" + "_S_r_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'rb') as f:
                S_r = pickle.load(f)
            with open("pilot/AMPNet" + "_S_i_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'rb') as f:
                S_i = pickle.load(f)
            if train_S == "Yes":
                self.S_r = nn.Parameter(S_r, requires_grad=True)
                self.S_i = nn.Parameter(S_i, requires_grad=True)
            else:
                self.register_buffer("S_r", S_r)
                self.register_buffer("S_i", S_i)

        elif pilot == "random":
            if train_S == "Yes":
                self.S_r = nn.Parameter(torch.randn(self.L, self.N), requires_grad=True)
                self.S_i = nn.Parameter(torch.randn(self.L, self.N), requires_grad=True)
            else:
                self.register_buffer("S_r", torch.randn(self.L, self.N))
                self.register_buffer("S_i", torch.randn(self.L, self.N))

        elif pilot == "zc":
            S = zc_sequence(self.L)[torch.randperm(self.L * (self.L - 1))[0:self.N], :].t()
            if train_S == "Yes":
                self.S_r = nn.Parameter(S.real, requires_grad=True)
                self.S_i = nn.Parameter(S.imag, requires_grad=True)
            else:
                self.register_buffer("S_r", S.real)
                self.register_buffer("S_i", S.imag)


    def forward(self, a_, g, H, Z):
        # g = g * 0.0000000001
        a = g - g
        batch_size = a.shape[0]
        norm = torch.sqrt(torch.sum(self.S_r ** 2 + self.S_i ** 2, dim=0) / self.L)
        S_r = torch.unsqueeze(self.S_r / norm, 0).expand(batch_size, -1, -1)
        S_i = torch.unsqueeze(self.S_i / norm, 0).expand(batch_size, -1, -1)
        S_rAG = S_r * torch.unsqueeze(a_, 1) * torch.unsqueeze(torch.sqrt(g), 1)
        S_iAG = S_i * torch.unsqueeze(a_, 1) * torch.unsqueeze(torch.sqrt(g), 1)
        Y_r = torch.bmm(S_rAG, H.real) - torch.bmm(S_iAG, H.imag) + Z.real
        Y_i = torch.bmm(S_iAG, H.real) + torch.bmm(S_rAG, H.imag) + Z.imag
        Sigma_hat_r = (torch.bmm(Y_r, torch.transpose(Y_r, 1, 2)) +
                       torch.bmm(Y_i, torch.transpose(Y_i, 1, 2))) / self.M
        Sigma_hat_i = (torch.bmm(Y_i, torch.transpose(Y_r, 1, 2)) -
                       torch.bmm(Y_r, torch.transpose(Y_i, 1, 2))) / self.M
        if self.pathloss == "U":
            g_ = g
            g = g / g
        for i in range(self.num_layer):

            S_rag = S_r * torch.unsqueeze(a, 1) * torch.unsqueeze(g, 1)
            S_iag = S_i * torch.unsqueeze(a, 1) * torch.unsqueeze(g, 1)
            Sigma_r = (torch.bmm(S_rag, torch.transpose(S_r, 1, 2)) +
                       torch.bmm(S_iag, torch.transpose(S_i, 1, 2)) +
                       self.sigma2 * torch.unsqueeze(self.I, 0))
            Sigma_i = (torch.bmm(S_iag, torch.transpose(S_r, 1, 2)) -
                       torch.bmm(S_rag, torch.transpose(S_i, 1, 2)))
            A_inv = torch.linalg.inv(Sigma_r)

            Sigma_inv_r = torch.linalg.inv(Sigma_r + torch.bmm(torch.bmm(Sigma_i, A_inv), Sigma_i))
            Sigma_inv_i = -torch.bmm(torch.bmm(A_inv, Sigma_i), Sigma_inv_r)
            Sigma_inv__S_r = torch.bmm(Sigma_inv_r, S_r) - torch.bmm(Sigma_inv_i, S_i)
            Sigma_inv__S_i = torch.bmm(Sigma_inv_i, S_r) + torch.bmm(Sigma_inv_r, S_i)
            SH__Sigma_inv__S___diag = torch.sum(S_r * Sigma_inv__S_r + S_i * Sigma_inv__S_i, dim=1)
            Sigma_hat__Sigma_inv__S_r = torch.bmm(Sigma_hat_r, Sigma_inv__S_r) - torch.bmm(Sigma_hat_i, Sigma_inv__S_i)
            Sigma_hat__Sigma_inv__S_i = torch.bmm(Sigma_hat_i, Sigma_inv__S_r) + torch.bmm(Sigma_hat_r, Sigma_inv__S_i)
            SH__Sigma_inv__Sigma_hat__Sigma_inv__S___diag = torch.sum(Sigma_inv__S_r * Sigma_hat__Sigma_inv__S_r +
                                                                      Sigma_inv__S_i * Sigma_hat__Sigma_inv__S_i, dim=1)
            gradient = ((SH__Sigma_inv__Sigma_hat__Sigma_inv__S___diag - SH__Sigma_inv__S___diag + 1e-5) /
                        (1e-5 + g * torch.pow(SH__Sigma_inv__S___diag, 2)))
            if self.pathloss == "K":
                d = torch.minimum(1 - a, torch.maximum(-a, gradient))
            elif self.pathloss == "U":
                d = torch.minimum(torch.max(g_) - a, torch.maximum(-a, gradient))
            # d = torch.minimum(1 - a.real, torch.maximum(-a.real, gradient))
            # print(torch.log(torch.linalg.det(torch.complex(Sigma_r, Sigma_i)/100) + 41 * math.log(100)).real)
            # print(torch.log(torch.sum(Sigma_inv_r * Sigma_hat_r + Sigma_inv_i * Sigma_hat_i, dim=1).sum()))
            # print(torch.log(torch.linalg.det(torch.complex(Sigma_r, Sigma_i)/100) + 41 * math.log(100)).real +
            #                 torch.log(torch.sum(Sigma_inv_r * Sigma_hat_r + Sigma_inv_i * Sigma_hat_i, dim=1).sum()))
            tmp = a
            a = a + self.gamma[i] * d

        #     print(torch.norm(a[0, :]-tmp[0, :]))
        #     Sigma_inv = torch.complex(Sigma_inv_r[0, :, :], Sigma_inv_i[0, :, :])
        #     Y = torch.complex(Y_r[0, :, :], Y_i[0, :, :]).cuda()
        #     tmp = torch.linalg.det(torch.complex(Sigma_r[0, :, :], Sigma_i[0, :, :]).cuda() + self.sigma2 * torch.eye(self.L).cuda())
        #     # if torch.isnan(tmp):
        #     #     tmp = torch.tensor(1e100).cuda()
        #
        # print('=' * 50)
        print(self.gamma)
        return a


class PSCANetMAP(nn.Module):
    def __init__(self, N, L, M, sigma2, pilot, train_S, train_gamma, pathloss, c, train_c, num_layer=20):
        super(PSCANetMAP, self).__init__()
        self.N = N
        self.L = L
        self.M = M
        self.num_layer = num_layer
        self.sigma2 = sigma2
        self.pathloss = pathloss
        self.register_buffer("I", torch.eye(self.L))

        tmp = torch.ones(num_layer) * 0.5
        for i in range(0, num_layer - 1):
            tmp[i + 1] = tmp[i] * (1 - 0.5 * tmp[i])

        if train_gamma == "Yes":
            self.gamma = nn.Parameter(tmp, requires_grad=True)
        else:
            self.register_buffer("gamma", 0.05 * torch.ones(self.num_layer))

        if pilot == "pretrain":
            with open("pilot/AMPNet" + "_S_r_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'rb') as f:
                S_r = pickle.load(f)
            with open("pilot/AMPNet" + "_S_i_N" + str(N) + "_L" + str(L) + ".pkl.bz2", 'rb') as f:
                S_i = pickle.load(f)
            if train_S == "Yes":
                self.S_r = nn.Parameter(S_r, requires_grad=True)
                self.S_i = nn.Parameter(S_i, requires_grad=True)
            else:
                self.register_buffer("S_r", S_r)
                self.register_buffer("S_i", S_i)

        elif pilot == "random":
            if train_S == "Yes":
                self.S_r = nn.Parameter(torch.randn(self.L, self.N), requires_grad=True)
                self.S_i = nn.Parameter(torch.randn(self.L, self.N), requires_grad=True)
            else:
                self.register_buffer("S_r", torch.randn(self.L, self.N))
                self.register_buffer("S_i", torch.randn(self.L, self.N))

        elif pilot == "zc":
            S = zc_sequence(self.L)[torch.randperm(self.L * (self.L - 1))[0:self.N], :].t()
            if train_S == "Yes":
                self.S_r = nn.Parameter(S.real, requires_grad=True)
                self.S_i = nn.Parameter(S.imag, requires_grad=True)
            else:
                self.register_buffer("S_r", S.real)
                self.register_buffer("S_i", S.imag)

        c = c.strip('[').strip(']').split(',')
        c = torch.tensor(list(map(float, c)))
        if train_c == "Yes":
            self.c = nn.Parameter(c / M, requires_grad=True)
            # tmp = torch.zeros_like(c)
            # self.c = nn.Parameter(tmp, requires_grad=True)
        elif train_c == "No":
            self.register_buffer("c", c / M)

            #             self.group_size = self.c.shape[0]
            #             if self.group_size == 1:
            #                self.index = []
            #            else:
        self.index = torch.stack([torch.arange(1, self.N, 2, dtype=torch.int),
                                  torch.arange(0, self.N, 2, dtype=torch.int)], dim=1).view(self.N)

    #     self.index = self.mvb_derivate_index(self.N, self.group_size)
    #
    # def mvb_derivate_index(self, N, group_size):
    #     tmp = [list(combinations(sorted(list(range(i * group_size, (i + 1) * group_size)), reverse=True),
    #     group_size - 1))
    #     for i in range(int(N / group_size))]
    #     index = [torch.tensor(list(combinations(tmp[k][j], i))) for k in range(int(N / group_size))
    #     for j in range(group_size)
    #     for i in range(1, group_size)]
    #     return index
    #
    # def mvb_derivate(self, a, index):
    #     if len(index) == 0:
    #         p = self.c[0]
    #     else:
    #         tmp = [a[:, index[i]].prod(axis=2).sum(axis=1) for i in range(len(index))]
    #         prod = torch.stack(tmp, dim=1)
    #         p = self.c[0] + (self.c[1:].reshape(1, self.group_size - 1, 1) *
    #                 prod.reshape(prod.shape[0], -1, self.N)).sum(axis=1)
    #
    #     return p

    # def mvb_derivate(self, a, index, c):
    #     # if len(index) == 0:
    #     #     p = c[0]
    #     # else:
    #     #     p = c[0] + c[1] * a[:, index]
    #     p = c[0] + c[1] * a[:, index]
    #
    #     return p

    def forward(self, a_, g, H, Z):

        a = g - g
        batch_size = a.shape[0]
        norm = torch.sqrt(torch.sum(self.S_r ** 2 + self.S_i ** 2, dim=0) / self.L)
        S_r = torch.unsqueeze(self.S_r / norm, 0).expand(batch_size, -1, -1)
        S_i = torch.unsqueeze(self.S_i / norm, 0).expand(batch_size, -1, -1)
        S_rAG = S_r * torch.unsqueeze(a_, 1) * torch.unsqueeze(torch.sqrt(g), 1)
        S_iAG = S_i * torch.unsqueeze(a_, 1) * torch.unsqueeze(torch.sqrt(g), 1)
        Y_r = torch.bmm(S_rAG, H.real) - torch.bmm(S_iAG, H.imag) + Z.real
        Y_i = torch.bmm(S_iAG, H.real) + torch.bmm(S_rAG, H.imag) + Z.imag
        Sigma_hat_r = (torch.bmm(Y_r, torch.transpose(Y_r, 1, 2)) +
                       torch.bmm(Y_i, torch.transpose(Y_i, 1, 2))) / self.M
        Sigma_hat_i = (torch.bmm(Y_i, torch.transpose(Y_r, 1, 2)) -
                       torch.bmm(Y_r, torch.transpose(Y_i, 1, 2))) / self.M
        if self.pathloss == "U":
            g_ = g
            g = g / g
        for i in range(self.num_layer):

            S_rag = S_r * torch.unsqueeze(a, 1) * torch.unsqueeze(g, 1)
            S_iag = S_i * torch.unsqueeze(a, 1) * torch.unsqueeze(g, 1)
            Sigma_r = (torch.bmm(S_rag, torch.transpose(S_r, 1, 2)) +
                       torch.bmm(S_iag, torch.transpose(S_i, 1, 2)) +
                       self.sigma2 * torch.unsqueeze(self.I, 0))
            Sigma_i = (torch.bmm(S_iag, torch.transpose(S_r, 1, 2)) -
                       torch.bmm(S_rag, torch.transpose(S_i, 1, 2)))
            A_inv = torch.linalg.inv(Sigma_r)
            Sigma_inv_r = torch.linalg.inv(Sigma_r + torch.bmm(torch.bmm(Sigma_i, A_inv), Sigma_i))
            Sigma_inv_i = -torch.bmm(torch.bmm(A_inv, Sigma_i), Sigma_inv_r)

            Sigma_inv__S_r = torch.bmm(Sigma_inv_r, S_r) - torch.bmm(Sigma_inv_i, S_i)
            Sigma_inv__S_i = torch.bmm(Sigma_inv_i, S_r) + torch.bmm(Sigma_inv_r, S_i)
            SH__Sigma_inv__S___diag = torch.sum(S_r * Sigma_inv__S_r + S_i * Sigma_inv__S_i, dim=1)
            Sigma_hat__Sigma_inv__S_r = torch.bmm(Sigma_hat_r, Sigma_inv__S_r) - torch.bmm(Sigma_hat_i, Sigma_inv__S_i)
            Sigma_hat__Sigma_inv__S_i = torch.bmm(Sigma_hat_i, Sigma_inv__S_r) + torch.bmm(Sigma_hat_r, Sigma_inv__S_i)
            SH__Sigma_inv__Sigma_hat__Sigma_inv__S___diag = torch.sum(Sigma_inv__S_r * Sigma_hat__Sigma_inv__S_r +
                                                                      Sigma_inv__S_i * Sigma_hat__Sigma_inv__S_i, dim=1)
            if i < 0:
                C = 0
            else:
                C = self.c[0] + self.c[1] * a[:, self.index]
            # C = self.mvb_derivate(a, self.index, c)
            if self.pathloss == "K":
                gradient = ((g * (SH__Sigma_inv__Sigma_hat__Sigma_inv__S___diag - SH__Sigma_inv__S___diag) + 1e-5 + C) /
                        (1e-5 + torch.pow(g * SH__Sigma_inv__S___diag, 2)))

                d = torch.minimum(1 - a, torch.maximum(-a, gradient))
            elif self.pathloss == "U":
                gradient = ((g * (SH__Sigma_inv__Sigma_hat__Sigma_inv__S___diag - SH__Sigma_inv__S___diag) + 1e-5 - self.M * p_upsilon_ratio(a)) /
                            (1e-5 + torch.pow(g * SH__Sigma_inv__S___diag, 2)))

                d = torch.minimum(torch.max(g_) - a, torch.maximum(-a, gradient))
            # d = torch.minimum(1 - a, torch.maximum(-a, gradient))
            tmp = a
            a = a + self.gamma[i] * d

            # print(torch.norm(a[0, :] - tmp[0, :]))

        print('=' * 50)
        return a

