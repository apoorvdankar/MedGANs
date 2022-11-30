import numpy as np

from PIL import Image
from PIL import ImageFilter

import cv2
from .EdgeEnhancer import *

image = cv2.imread('test.png')

edgeEnhancer = EdgeEnhancer()
edgeEnhancer.process(image)

