import cv2
import matplotlib.pyplot as plt

from hdr import HDR
from hdr_utils import show_origin_and_output

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