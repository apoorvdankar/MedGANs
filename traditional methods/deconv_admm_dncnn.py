import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf

# PyTorch
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps")
from network_dncnn import DnCNN as net


def deconv_admm_dncnn(b, c, lam, rho, num_iters):

    # Blur kernel
    cFT = psf2otf(c, b.shape)
    cTFT = np.conj(cFT)

    # Fourier transform of b
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)

    # set up DnCNN model
    n_channels = 1
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=17, act_mode='R')
    model.load_state_dict(torch.load('dncnn_25.pth'), strict=True)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    ################# begin task 2 ###################################

    # ADMM with DnCNN doesn't require a gradient function, so we don't
    # need it here. Just pre-compute the denominator for the x-update
    # here, because that doesn't change unless rho changes, which is
    # not the case here

    # pre-compute denominator of x update
    denom = cTFT * cFT + rho  # you need to edit this placeholder

    ################# end task 2 ####################################

    for it in tqdm(range(num_iters)):

        ################# begin task 2 ###################################

        # Complete this part by implementing the x-update discussed in
        # class and in the problem session. If you implemented the
        # denominator term above, you only need to compute the nominator
        # here as well as the rest of the x-update

        # x update - inverse filtering: Fourier multiplications and divisions
        v = z - u
        nomi = cTFT * bFT + rho*np.fft.fft2(v)
        x = np.real(np.fft.ifft2(nomi / denom))
        ################# end task 2 ####################################

        # z update
        v = x + u

        # run DnCNN denoiser
        v_tensor = torch.reshape(torch.from_numpy(v).float().to(device), (1, 1, v.shape[0], v.shape[1]))
        v_tensor_denoised = model(v_tensor)
        z = torch.squeeze(v_tensor_denoised).cpu().numpy()

        # u update
        u = u + x - z

    return x
