import torch 
from torch.utils.data import DataLoader
from models import *
from const import *
from denoiser_utils import *
from HDR.hdr import *
from HDR.hdr_utils import *

# PART 1 OF PIPELINE: DENOISING 

# Loading Denoisers from CKPT file
randomDenoiser = load_generator_from_ckpt(ckpt_random)
poissonDenoiser = load_generator_from_ckpt(ckpt_poisson)

# Loading Test Data batch to be tested
real_images, noisy_images = create_tensor_data(test_data_dir)

# Creating the test_batch
batch_size = 4
test_data = torch.stack((real_images,noisy_images), axis=1)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = False)
dataiter = iter(test_loader)
test_batch = dataiter.next()

# Real and Noisy images
real_batch = test_batch[:,0].to(device)
noisy_batch = test_batch[:,1].to(device)
noisy_image = noisy_batch.to(device)

# Running the Denoisers on the noisy image to get denoised images
random_denoiser_output = randomDenoiser(noisy_image)
poisson_denoiser_output = poissonDenoiser(random_denoiser_output)

# Uncomment the following line to see the outputof denoisers
# show_tensor_images(poisson_denoiser_output, name='poisson_output')

# Caluculating the metrics of the denoised images
print('-'*50)
print('Calculating the metrics of denoised Images')
print('-'*50)

noisy_image_psnr = calculate_batch_psnr(noisy_image, real_batch)
print('Noisy Image PSNR: ', noisy_image_psnr.item())

random_denoiser_output_psnr = calculate_batch_psnr(random_denoiser_output, real_batch)
print('Denoised(Random) Image PSNR: ', random_denoiser_output_psnr.item())

poisson_denoiser_output_psnr = calculate_batch_psnr(poisson_denoiser_output, real_batch)
print('Denoised(Poisson) Image PSNR: ', poisson_denoiser_output_psnr.item())

print('-'*50)

noisy_image_simi = calculate_batch_ssim(noisy_image, real_batch)
print('Noisy Image SSIM: ', noisy_image_simi.item())

random_denoiser_output_simi = calculate_batch_ssim(random_denoiser_output, real_batch)
print('Denoised(Random) Image SSIM: ', random_denoiser_output_simi.item())

poisson_denoiser_output_simi = calculate_batch_ssim(poisson_denoiser_output, real_batch)
print('Denoised(Poisson) Image SSIM: ', poisson_denoiser_output_simi.item())

print('-'*50)

# PART 2 OF PIPELINE: HDR CONVERSION

# Loading the HRD Class

hdr_handler = HDR(True)
hdr_output = hdr_handler.process_tensor_batch(poisson_denoiser_output)

hdr_output_psnr = calculate_batch_psnr(hdr_output, real_batch)
print('HDR Image PSNR: ', hdr_output_psnr.item())

hdr_output_simi = calculate_batch_ssim(hdr_output, real_batch)
print('HDR Image SSIM: ', hdr_output_simi.item())

# SHowing Final Output of the pipeline 

show_tensor_images(hdr_output, name='hdr_output')
