import torch
from .hdr_utils import *

# The HDR method is an adaptation of a research paper given below: 
'''
J. S. Park, J. W. Soh, and N. I. Cho, Generation of High
Dynamic Range Illumination from a Single Image for the Enhancement
of Undesirably Illuminated Images. USA: Kluwer Academic
Publishers, jul 2019, vol. 78, no. 14. [Online]. Available:
https://doi.org/10.1007/s11042-019-7384-z
'''

class HDR():

    def __init__(self, flag):
        self.weighted_fusion = flag
        self.wls = wls_decompositon
        self.reflectance_scaling = reflectance_scaling
        self.vig = generate_illuminations
        self.tonemap = tonemapping

    def process(self, image):

        if image.shape[2] == 4:
            image = image[:,:,0:3]
        S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
        image = 1.0*image/255
        L = 1.0*S

        I = self.wls(S)
        R = np.log(L+1e-22) - np.log(I+1e-22)
        R_ = self.reflectance_scaling(R, L)
        I_K = self.vig(L, 1.0 - L)

        result_ = self.tonemap(image, L, R_, I_K, self.weighted_fusion)
        return result_

    def process_tensor_batch(self, batch):
        batch_np = batch.detach().numpy()

        hdr_batch = torch.zeros([len(batch_np), 1, 128, 128], dtype=torch.float32).to('cpu')

        for image_index in range(len(batch_np)):
            image = batch_np[image_index].reshape((128,128))
            
            rgb_image = np.stack((image,)*3, axis=-1)
            upscaled_image = cv2.convertScaleAbs(rgb_image, alpha=(255.0))
            
            output_image = self.process(upscaled_image) 
            
            output_image = output_image[:, :, 0].reshape((1, 128, 128))
            
            hdr_batch[image_index] = torch.Tensor(output_image)

        return hdr_batch