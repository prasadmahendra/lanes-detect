import logging
import os
import cv2
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from image_processing import ImageProcessing
from calibrate import Calibrate

class PerspectiveTransform(ImageProcessing):
    def __init__(self, config, load_saved_trans_matrix=False):
        super(PerspectiveTransform, self).__init__(config)
        self.__config = config
        self.__output_images_dir = config.get('detection', 'output_directory')
        self.__data_dir = config.get('camera_calibration', 'data_directory')
        self.__test_dir = config.get('test', 'test_images_directory')
        self.__load_saved_trans_matrix = load_saved_trans_matrix
        self.__data_loaded = False
        self.__M = None
        self.__Minv = None

        if self.__load_saved_trans_matrix:
            self.init()

    def selfdiag(self):
        self.init()

    def init(self, image=None):
        save_path = "{0}/results/perspective_trans/trans_matrix.pickle".format(self.__data_dir)
        if self.__load_saved_trans_matrix == False:
            path = "{0}/{1}".format(self.__test_dir, "straight_lines1.jpg")

            if not os.path.isfile(path):
                raise FileNotFoundError("{0} not found!".format(path))

            image = self.load_image(path)
            cam_calib = Calibrate(self.__config)
            corrected = cam_calib.undistort(image)

            color = [255, 0, 0]
            thickness = 5

            x_min = 280
            x_max = 1045
            src = np.float32([[x_min, 675], [575, 465], [x_max, 675], [715, 465]])
            dst = np.float32([[x_min, 675], [x_min, 0], [x_max, 675], [x_max, 0]])

            self.__M = cv2.getPerspectiveTransform(src, dst)
            self.__Minv = cv2.getPerspectiveTransform(dst, src)
            img_size = (corrected.shape[1], corrected.shape[0])

            warped = cv2.warpPerspective(corrected, self.__M, img_size, flags=cv2.INTER_LINEAR)
            warped_inv = cv2.warpPerspective(warped, self.__Minv, img_size, flags=cv2.INTER_LINEAR)

            cv2.line(warped, tuple(dst[0]), tuple(dst[1]), color, thickness)
            cv2.line(warped, tuple(dst[2]), tuple(dst[3]), color, thickness)

            pickle.dump([self.__M, self.__Minv], open(save_path, "wb"))

            cv2.line(corrected, tuple(src[0]), tuple(src[1]), color, thickness)
            cv2.line(corrected, tuple(src[2]), tuple(src[3]), color, thickness)

            self.display_image_grid('perspective', 'perspective_correction.jpg', [image, corrected, warped, warped_inv], ['image', 'corrected', 'warped', 'warped_inv'], save=self.save_output_images())
        else:
            if self.__data_loaded == False:
                self.__M, self.__Minv = pickle.load(open(save_path, 'rb'))
                self.__data_loaded = True

        return image

    def process(self, image, filename=None, display=False):
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, self.__M, img_size, flags=cv2.INTER_LINEAR)
        return warped

    def get_minv(self):
        return self.__Minv

    def get_m(self):
        return self.__M

