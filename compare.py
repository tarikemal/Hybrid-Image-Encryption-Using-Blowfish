import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compareImagesWithSSIM(imageA, imageB):
    # Calculate the SSIM
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    s = ssim(gray1, gray2)
    # Return the SSIM. The higher the value, the more "similar" the two images are.
    return s

def compareImagesWithPSNR(imageA, imageB):
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB)
    
    comp = cv2.PSNR(image1, image2)
    return comp
