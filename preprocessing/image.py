import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def resize(self, width=None, height=None):
        if width is None and height is None:
            return

        if width is not None and height is not None:
            self.image = cv2.resize(self.image, (width, height))
        elif width is not None:
            height = int(self.image.shape[0] * (width / self.image.shape[1]))
            self.image = cv2.resize(self.image, (width, height))
        else:
            width = int(self.image.shape[1] * (height / self.image.shape[0]))
            self.image = cv2.resize(self.image, (width, height))

    def to_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur(self, ksize=(5, 5)):
        self.image = cv2.GaussianBlur(self.image, ksize, 0)

    def threshold(self, thresh=128, maxval=255, type=cv2.THRESH_BINARY):
        _, self.image = cv2.threshold(self.image, thresh, maxval, type)

    def canny(self, threshold1=100, threshold2=200):
        self.image = cv2.Canny(self.image, threshold1, threshold2)

    def save(self, output_path=None):
        if output_path is None:
            output_path = self.image_path

        cv2.imwrite(output_path, self.image)

