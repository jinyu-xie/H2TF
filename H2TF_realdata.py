from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.optim
from model import *
from utils import *

dtype = torch.cuda.FloatTensor  # type: ignore


def closure(iter):
    global psnr_best, ssim_best, count, clean_image, t1, out_last

    out_ = net()

    D_x_, D_y_ = TV(out_)
    D_xz_, D_yz_ = SSTV(out_)
    D_x = D_x_.clone().detach()
    D_y = D_y_.clone().detach()
    D_xz = D_xz_.clone().detach()
    D_yz = D_yz_.clone().detach()
    out = out_.clone().detach()

    if iter == 0:
        global S, mu, thres_s, thres_tv, thres_sstv, D_2, D_3, D_4, D_5, V_2, V_3, V_4, V_5

        D_2 = torch.zeros([img_noisy.shape[0] - 1, img_noisy.shape[1], img_noisy.shape[2]]).type(dtype)
        D_3 = torch.zeros([img_noisy.shape[0], img_noisy.shape[1] - 1, img_noisy.shape[2]]).type(dtype)
        D_4 = torch.zeros([img_noisy.shape[0] - 1, img_noisy.shape[1], img_noisy.shape[2] - 1]).type(dtype)
        D_5 = torch.zeros([img_noisy.shape[0], img_noisy.shape[1] - 1, img_noisy.shape[2] - 1]).type(dtype)

        V_2 = D_x.type(dtype)
        V_3 = D_y.type(dtype)
        V_4 = D_xz.type(dtype)
        V_5 = D_yz.type(dtype)

        S = (img_noisy - out).type(dtype)

    S = soft_thres(img_noisy - out, thres_s)

    V_2 = soft_thres(D_x + D_2 / mu, thres_tv)
    V_3 = soft_thres(D_y + D_3 / mu, thres_tv)

    V_4 = soft_thres(D_xz + D_4 / mu, thres_sstv)
    V_5 = soft_thres(D_yz + D_5 / mu, thres_sstv)

    total_loss = mu / 2 * torch.norm(D_x_ - (V_2 - D_2 / mu), 2)
    total_loss += mu / 2 * torch.norm(D_y_ - (V_3 - D_3 / mu), 2)
    total_loss += 10 * mu / 2 * torch.norm(D_xz_ - (V_4 - D_4 / mu), 2)
    total_loss += 10 * mu / 2 * torch.norm(D_yz_ - (V_5 - D_5 / mu), 2)
    total_loss += torch.norm(img_noisy - out_ - S, 2)
    # total_loss += torch.norm((img_noisy - out_) * mask - S, 2)

    total_loss.backward()

    D_2 = (D_2 + mu * (D_x - V_2)).clone().detach()
    D_3 = (D_3 + mu * (D_y - V_3)).clone().detach()
    D_4 = (D_4 + mu * (D_xz - V_4)).clone().detach()
    D_5 = (D_5 + mu * (D_yz - V_5)).clone().detach()

    out_np = out.detach().cpu().squeeze().numpy()

    if iter % (show_every * 20) == 0:
        print("iter", iter)
        plt.figure(figsize=(11, 22))
        plt.subplot(121)
        plt.imshow(np.clip(np.stack((img_noisy_np[:, :, show[0]],
                                     img_noisy_np[:, :, show[1]],
                                     img_noisy_np[:, :, show[2]]), 2), 0, 1))
        plt.title("Observed")
        plt.subplot(122)
        plt.imshow(np.clip(np.stack((out_np[:, :, show[0]],
                                     out_np[:, :, show[1]],
                                     out_np[:, :, show[2]]), 2), 0, 1))
        plt.title("Recovered")
        plt.show()
    if iter == num_iter - 1:
        print(dataname, "r", r, "mu", mu, "Thres", Thres, "time", time.time() - t0)

    return 0, 0


show_every = 10
num_iter = 1001
lr = 0.005
show = [10, 20, 30]

Thres = [
    # [0.02, 0.2, 0.2],
    # [0.04, 0.2, 0.2],
    [0.2, 0.2, 0.2],
    # [0.4, 0.2, 0.2],
]
mu = 0.15  # 0.05, 0.07, 0.09, 0.1, 0.15, 0.17, 0.19
r = [
    # [80, 40, 20, 10],
    # [72, 36, 18, 9],
    # [64, 32, 16, 8],
    # [56, 28, 14, 7],
    # [48, 24, 12, 6],
    [40, 20, 10, 5],
    # [32, 16, 8, 4],
    # [24, 12, 6, 3],
    # [16, 8, 4, 2],
]
datapath = "./realdata/"
dataname = "urban"

mat = scipy.io.loadmat(datapath + dataname)
img_noisy_np = mat["Nhsi"]
img_noisy = torch.from_numpy(img_noisy_np).type(dtype).cuda()

mask = torch.zeros(img_noisy.shape).type(dtype)
for i in range(mask.shape[1]):
    for j in range(mask.shape[2]):
        if img_noisy_np[:, i, j].sum() != 0:
            mask[:, i, j] = 1

setup_seed(0)

t0 = time.time()
thres_s, thres_tv, thres_sstv = Thres[0]

net = H2TF(img_size=img_noisy_np.shape, r=r[0], m=2).to(device)
soft_thres = soft()
TV = TV_Loss()
SSTV = SSTV_Loss()

p = [x for x in net.parameters()]
total_num = sum(pp.numel() for pp in p)
print('total param. is ', total_num*4/1024/1024, ' Mb')

optimize(p, closure, lr, num_iter)
