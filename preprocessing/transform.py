import numpy as np
import cv2

class Transform:
    def __init__(self, src_pts, dst_pts):
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def apply(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    def apply_inverse(self, img):
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.M_inv, img_size, flags=cv2.INTER_LINEAR)

        return unwarped
