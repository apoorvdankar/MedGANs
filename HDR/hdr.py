import torch
from hdr_utils import *

class HDR():

    def __init__(self, flag):
        self.weighted_fusion = flag
        self.wls = wlsFilter
        self.srs = SRS
        self.vig = VIG
        self.tonemap = tonereproduct

    def process(self, image):

        if image.shape[2] == 4:
            image = image[:,:,0:3]
        S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
        image = 1.0*image/255
        L = 1.0*S

        I = self.wls(S)
        R = np.log(L+1e-22) - np.log(I+1e-22)
        R_ = self.srs(R, L)
        I_K = self.vig(L, 1.0 - L)

        result_ = self.tonemap(image, L, R_, I_K, self.weighted_fusion)
        return result_

    def process_tensor_batch(self, batch):
        batch_np = batch.detach().numpy()

        hdr_batch = torch.zeros(len(batch_np), 1, 128, 128).to('cpu')

        for image_index in range(len(batch_np)):
            image = batch_np[image_index].reshape((128,128))
            
            rgb_image = np.stack((image,)*3, axis=-1)
            upscaled_image = cv2.convertScaleAbs(rgb_image, alpha=(255.0))
            
            output_image = self.process(upscaled_image) 
            
            output_image = output_image[:, :, 0].reshape((1, 128, 128))
            
            hdr_batch[image_index] = torch.Tensor(output_image)

        return hdr_batch