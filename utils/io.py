import os
import cv2


def read_image(file_path):
    """Reads an image from file.

    Args:
        file_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    return cv2.imread(file_path)


def save_image(image, file_path):
    """Saves an image to file.

    Args:
        image (numpy.ndarray): The image as a NumPy array.
        file_path (str): The path to save the image to.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, image)
