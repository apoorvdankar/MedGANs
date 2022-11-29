# CONST file
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 10

size = 128, 128

test_data_dir = './test_data/'

ckpt_random = './ckpt/random_generator.pth'

ckpt_poisson = './ckpt/poisson_generator.pth'