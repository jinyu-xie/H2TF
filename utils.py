import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio


class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :-1, :] - a[:, 1:, :])
        gradient_a_y = torch.abs(a[:-1, :, :] - a[1:, :, :])
        return gradient_a_y, gradient_a_x


class SSTV_Loss(nn.Module):
    def __init__(self):
        super(SSTV_Loss, self).__init__()

    def forward(self, a):
        gradient_a_z = torch.abs(a[:, :, :-1] - a[:, :, 1:])
        gradient_a_yz = torch.abs(gradient_a_z[:-1, :, :] - gradient_a_z[1:, :, :])
        gradient_a_xz = torch.abs(gradient_a_z[:, :-1, :] - gradient_a_z[:, 1:, :])
        return gradient_a_yz, gradient_a_xz


class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()

    def forward(self, x, lam):
        x_abs = x.abs() - lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out


def psnr3d(x, y):
    p = 0
    for i in range(x.shape[2]):
        p_he = peak_signal_noise_ratio(np.clip(x[:, :, i], 0, 1), np.clip(y[:, :, i], 0, 1))
        p += p_he

    return p/x.shape[2]


def optimize(parameters, closure, LR, num_iter, error=-1):
    optimizer = torch.optim.Adam(parameters, lr=LR)
    if error == -1:
        for j in range(num_iter):
            optimizer.zero_grad()
            _, _ = closure(j)
            optimizer.step()
    else:
        for j in range(num_iter):
            optimizer.zero_grad()
            _, _, err = closure(j)
            if err <= error:
                break
            optimizer.step()
