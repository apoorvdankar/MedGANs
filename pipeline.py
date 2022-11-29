import torch 
from torch.utils.data import DataLoader
from models import *
from const import *
from denoiser_utils import *

randomDenoiser = Generator()

randomGenckpt = torch.load(ckpt_random, map_location=torch.device('cpu'))
randomDenoiser.load_state_dict(randomGenckpt["Gen"])
randomDenoiser.to(device)
randomDenoiser.double()
randomDenoiser.eval()

poissonDenoiser = Generator()

poissonCheckpoint = torch.load(ckpt_random, map_location="cpu")
poissonDenoiser.load_state_dict(poissonCheckpoint["Gen"])
poissonDenoiser.to(device)
poissonDenoiser.double()
poissonDenoiser.eval()

real_images, noisy_images = create_tensor_data(test_data_dir)

batch_size=4

test_data = torch.stack((real_images,noisy_images), axis=1)
a_loader = DataLoader(test_data, batch_size=batch_size, shuffle = True)
noisy_image_loader = DataLoader(noisy_images, batch_size=batch_size, shuffle = True)
dataiter = iter(a_loader)

batch = dataiter.next()

batch0 = batch[:,0].to(device)
batch1 = batch[:,1].to(device)

noisy_image = batch1.to(device)
randomGen = randomDenoiser(noisy_image)
poissonGen = poissonDenoiser(randomGen)
show_tensor_images(poissonGen)

# show_tensor_images(batch0)

# show_tensor_images(randomGen)

# Functions defining Evaluation Metrics



fpass_psnr = calculate_batch_psnr(randomGen, batch0)
print(fpass_psnr)

spass_psnr = calculate_batch_psnr(poissonGen, batch0)
print(spass_psnr)

fpass_simi = calculate_batch_ssim(randomGen, batch0)
print(fpass_simi)

spass_simi = calculate_batch_ssim(poissonGen, batch0)
print(spass_simi)

# torch.save(poissonGen,"/content/drive/MyDrive/Denoising_GANs/output.pt")

