import cv2
import numpy as np

def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def resize_image(image, new_size):
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

def binarize_image(image, threshold=128):
    binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    return binary_image

def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image
