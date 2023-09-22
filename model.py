import math
import torch
import torch.nn as nn
import numpy as np
import random

dtype = torch.cuda.FloatTensor


def get_device():
    if torch.cuda.is_available():
        de = 'cuda:0'
    else:
        de = 'cpu'
    return de


device = get_device()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class H2TF(nn.Module):
    def __init__(self, img_size, r=(24, 18, 12, 6), m=2):
        super(H2TF, self).__init__()
        self.img_size = img_size
        self.r = r
        self.l = len(r) + 1
        self.m = m
        self.stdv = 1. / math.sqrt(self.img_size[2])
        self.rs = []
        self.act = []
        for i in range(self.l):
            if i == 0:
                self.rs.append(nn.Parameter(torch.Tensor(self.img_size[2], self.img_size[0], self.r[i])).to(device))
            elif i == (self.l - 1):
                self.rs.append(nn.Parameter(torch.Tensor(self.img_size[2], self.r[i-1], self.img_size[1])).to(device))
            else:
                self.rs.append(nn.Parameter(torch.Tensor(self.img_size[2], self.r[i-1], self.r[i])).to(device))
                self.act.append(nn.LeakyReLU())
        self.rs = nn.ParameterList(self.rs)

        self.hnt = [nn.Linear(self.img_size[2], self.img_size[2], bias=False)]
        for i in range(self.m - 1):
            self.hnt.append(nn.LeakyReLU())
            self.hnt.append(nn.Linear(self.img_size[2], self.img_size[2], bias=False))
        self.hnt = nn.Sequential(*self.hnt)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.rs)):
            self.rs[i].data.uniform_(-self.stdv, self.stdv)

    def forward(self):
        for i in range(self.l - 1):
            if i == 0:
                out = torch.matmul(self.rs[0], self.rs[1])
            else:
                out = torch.matmul(self.act[i-1](out), self.rs[i + 1])
        out = self.hnt(out.permute(1, 2, 0))
        return out
