import torch 
import numpy as np 
import os
from PIL import Image
from const import *
from tqdm.notebook import tqdm
from torchvision.utils import make_grid
import random
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
from models import *

import matplotlib.pyplot as plt 

def reduce_size(image):
    im_resized = image.resize(size, Image.LANCZOS)
    return np.array(im_resized).reshape(1,*size)/255


def create_noisy_image(noise_typ, image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**1.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss/10
        return noisy


def create_tensor_data(img_dir):
    real_images = []
    noisy_images = []
    for f in tqdm(os.listdir(img_dir)):
        if os.path.splitext(f)[1] not in ['.png', '.jpg', '.jpeg']:
            continue
        image = Image.open(img_dir + str(f))
        image = reduce_size(image)
        if len(image.shape)>3:
            continue
        real_images.append(image)
        noi = ["gauss","poisson","speckle"]
        random_noise = random.choice(noi)
        noisy_image = create_noisy_image(random_noise,image)
        noisy_images.append(noisy_image)
    return torch.tensor(np.array(real_images)).double(), torch.tensor(np.array(noisy_images)).double()


def show_tensor_images(tensor_img, num_images = 4, size=(1, 128, 128), name='img'):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=2)
    plt.figure(figsize=(7,7))
    
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f'./{name}.jpg')
    plt.show()

def calculate_batch_psnr(noisy_image,real_image):
    psnr = PeakSignalNoiseRatio().to(device)
    psnr_score = psnr(noisy_image, real_image)
    return psnr_score

def calculate_batch_ssim(noisy_image, real_image):
    noisy_image = noisy_image.to(torch.double)
    print(noisy_image.dtype, real_image.dtype)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ssim_score = ssim(noisy_image, real_image)
    return ssim_score

def load_generator_from_ckpt(ckpt_file_name):
    Denoiser = Generator()
    DenoiserCKPT = torch.load(ckpt_file_name, map_location=torch.device(device))
    Denoiser.load_state_dict(DenoiserCKPT["Gen"])
    Denoiser.to(device)
    Denoiser.double()
    Denoiser.eval()
    return Denoiser
