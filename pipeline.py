import logging
import os
import cv2
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from calibrate import Calibrate

class Pipeline(object):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.__cam_calibrate = Calibrate(config)

    def selfdiag(self):
        self.process(mpimg.imread('data/test_images/test1.jpg'))

    def process(self, image):
        image = self.undistort(image)
        image = self.color_grad_threshold(image)
        image = self.perspective_transform(image)

        self.__display_image(image)

    def undistort(self, image):
        return self.__cam_calibrate.undistort(image)

    def color_grad_threshold(self, image):
        # TODO
        return image

    def perspective_transform(self, image):
        # TODO
        return image

    def detect_lane_lines(self):
        #TODO
        pass

    def determine_curvature(self):
        #TODO
        pass

    def __display_image(self, image):
        plt.imshow(image)
        plt.show()
