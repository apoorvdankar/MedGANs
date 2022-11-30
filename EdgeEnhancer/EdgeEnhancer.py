import torch
import numpy as np
import cv2

class EdgeEnhancer():

    def __init__(self):
        pass

    def process(self, image, kernel_size, sigma):
        smoothness_mask = cv2.GaussianBlur(image, kernel_size, sigma) 
        edge_enhanced_image = cv2.addWeighted(image, 1.5, smoothness_mask, -0.45, 0)

        return edge_enhanced_image
    
    def process_tensor_batch(self, batch, kernel_size, sigma):
        batch_np = batch.detach().numpy()

        edge_enhanced_batch = torch.zeros(len(batch_np), 1, 128, 128).to('cpu')

        for image_index in range(len(batch_np)):
            image = batch_np[image_index].reshape((128,128))
            
            rgb_image = np.stack((image,)*3, axis=-1)
            
            output_image = self.process(rgb_image, kernel_size, sigma) 
            
            output_image = output_image[:, :, 0].reshape((1, 128, 128))
            
            edge_enhanced_batch[image_index] = torch.Tensor(output_image)

        return edge_enhanced_batch