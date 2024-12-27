import numpy as np
from scipy.fftpack import fft2, ifft2
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, input_size, eta):
        super(Generator, self).__init__()

        self.lambda_val_param = nn.Parameter(torch.rand(input_size) * eta, requires_grad=True)
        self.el_dif_generator = nn.Conv2d(102, 102, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x_fft = fft.fftn(x, dim=(-3, -2, -1))

        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        if amplitude.shape[0] < self.lambda_val_param.shape[0]:
            lambda_val = torch.nn.Parameter(self.lambda_val_param[:amplitude.shape[0], ...])
            interpolated_amplitude = lambda_val * amplitude + (1 - lambda_val) * amplitude.mean()

        else:
            lambda_val = self.lambda_val_param
            interpolated_amplitude = lambda_val * amplitude + (1 - lambda_val) * amplitude.mean()

        x_ED_fft = interpolated_amplitude * torch.exp(1j * phase)

        x_ED = fft.ifftn(x_ED_fft, dim=(-3, -2, -1)).real

        return x_ED


