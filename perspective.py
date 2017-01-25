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

class PerspectiveTransform(ImageProcessing):
    def __init__(self, config, load_saved_trans_matrix=False):
        super(PerspectiveTransform, self).__init__(config)
        self.__output_images_dir = config.get('detection', 'output_directory')
        self.__data_dir = config.get('camera_calibration', 'data_directory')
        self.__test_dir = config.get('test', 'test_images_directory')
        self.__load_saved_trans_matrix = False
        self.__data_loaded = False

    def selfdiag(self):
        self.transform()

    def transform(self, image=None):
        save_path = "{0}/results/perspective_trans/trans_matrix.pickle".format(self.__data_dir)
        if self.__load_saved_trans_matrix == False:
            path = "{0}/{1}".format(self.__test_dir, "straight_lines2.jpg")

            if not os.path.isfile(path):
                raise FileNotFoundError("{0} not found!".format(path))

            image = self.load_image(path)

            color = [255, 0, 0]
            thickness = 5

            # left lane
            (x1, y1) = [270, 675]
            (x2, y2) = [580, 465]

            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

            # horizon
            (x1, y1) = [580, 465]
            (x2, y2) = [710, 465]

            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

            # right lane
            (x1, y1) = [1035, 675]
            (x2, y2) = [710, 465]

            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

            src = np.float32([[270, 675], [580, 465], [1035, 675], [710, 465]])
            dst = np.float32([[270, 675], [270, 0], [1035, 675], [1035, 0]])

            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)
            img_size = (image.shape[1], image.shape[0])
            warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
            warped_inv = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)

            pickle.dump([M, Minv], open(save_path, "wb"))

            self.display_image_grid('perspective_correction.jpg', [image, warped, warped_inv], ['image', 'warped', 'warped_inv'], save=True)
        else:
            if self.__data_loaded == False:
                M, Minv = pickle.load(save_path)
                self.__data_loaded = True



        return image


