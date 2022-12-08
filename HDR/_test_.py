import cv2
import torch
import matplotlib.pyplot as plt

from hdr import HDR
from hdr_utils import show_origin_and_output


# Below function shouldn't be used in pipeline, they just explain how to apply HDR on single image or a batch

def HDR_Image():
    # Read image from file
    image = cv2.imread('test.png')

    # Create a instance of HDR Handler 
    HDR_Handler = HDR(True) 

    # Process the image using the HDR Handler
    output_image = HDR_Handler.process(image) 

    # Save and show the final result 
    cv2.imwrite('result.jpg', 255*output_image)

    # Get a preview of result using matplotlib
    show_origin_and_output(image, output_image)

# When applying HDR on a tensor batch of images
def HDR_Batch():
    batch = torch.load('output.pt', map_location = torch.device('cpu'))

    hdr_handler = HDR(True)
    hdr_batch = hdr_handler.process_tensor_batch(batch)