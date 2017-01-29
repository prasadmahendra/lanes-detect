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

class CalibrateImage(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config, fullpath, name, img_obj_pts_x, img_obj_pts_y, results_dir, output_images_dir):
        super(CalibrateImage, self).__init__(config)
        self.__fullpath = fullpath
        self.__name = name
        self.__img_obj_pts_x = img_obj_pts_x
        self.__img_obj_pts_y = img_obj_pts_y
        self.__results_dir = results_dir
        self.__output_images_dir = output_images_dir

    def name(self):
        return self.__name

    def calibrate(self):
        img = self.load_image(self.__fullpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (self.__img_obj_pts_x, self.__img_obj_pts_y), None)

        if ret == True:
            # Draw and display the corners
            img_with_corners_drawn = self.load_image(self.__fullpath)
            cv2.drawChessboardCorners(img_with_corners_drawn, (self.__img_obj_pts_x, self.__img_obj_pts_y), corners, ret)

            # generate object points based on number of chessboard (inside) corners
            objp = np.zeros((self.__img_obj_pts_x * self.__img_obj_pts_y, 3), np.float32)
            objp[:,:2] = np.mgrid[0:self.__img_obj_pts_x, 0:self.__img_obj_pts_y].T.reshape(-1, 2) # x,y object point coordinates

            objpoints = list()
            objpoints.append(objp)

            imgpoints = list()
            imgpoints.append(corners)

            # mtx: camera matrix
            # dist: distortion co-efficients
            # rvecs: rotation vector (rot position of the camera in the world)
            # tvecs: translation vector (trans position of the camera in the world)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # undistort the original image using the camera matrix and distortion co-efficients
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

            # save mtx, dist, rvecs, tvecs
            pickle_filename = '{0}/calibration_data/{1}.pickle'.format(self.__results_dir, self.name())
            pickle.dump([mtx, dist, rvecs, tvecs], open(pickle_filename, "wb"))

            self.display_image_grid("camera_cal", self.name(), [img, img_with_corners_drawn, undistorted_img], ['orig image', 'img_with_corners_drawn', 'undistorted_img'], cmap='gray', save=self.save_output_images())
        else:
            self.__logger.debug("chess board corners not found on {0} nx: {1} ny: {2}".format(self.name(), self.__img_obj_pts_x, self.__img_obj_pts_y))


class Calibrate(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        super(Calibrate, self).__init__(config)
        self.__config = config
        self.__data_dir = config.get('camera_calibration', 'data_directory')
        self.__results_dir = '{0}/results'.format(self.__data_dir)
        self.__output_images_dir = '{0}/camera_cal'.format(config.get('camera_calibration', 'output_directory'))
        self.__filename_fmt = config.get('camera_calibration', 'filename_fmt')
        self.__file_start_idx = config.getint('camera_calibration', 'files_start_index')
        self.__file_end_idx = config.getint('camera_calibration', 'files_end_index')
        self.__img_obj_pts_x = config.getint('camera_calibration', 'object_points_x')
        self.__img_obj_pts_y = config.getint('camera_calibration', 'object_points_y')

        self.__undistort_data_file = config.get('camera_calibration', 'undistort_data_file_to_use')
        self.__undistort_data = None
        pass

    def selfdiag(self):
        self.calibrate()
        self.undistort(mpimg.imread('data/test_images/test4.jpg'))

    def camera_images_iterator(self):
        i = self.__file_start_idx
        while i <= self.__file_end_idx:
            filename = str.format(self.__filename_fmt, i)
            yield CalibrateImage(self.__config, '{0}/{1}'.format(self.__data_dir, filename), filename, self.__img_obj_pts_x, self.__img_obj_pts_y, self.__results_dir, self.__output_images_dir)
            i += 1

    def calibrate(self):
        for calib_image in tqdm(self.camera_images_iterator(), total=1 + (self.__file_end_idx - self.__file_start_idx)):
            calib_image.calibrate()


    def undistort(self, img):
        if self.__undistort_data == None:
            if not os.path.isfile(self.__undistort_data_file):
                raise FileNotFoundError("Missing {0}".format(self.__undistort_data_file))
            self.__undistort_data = pickle.load(open(self.__undistort_data_file, 'rb'))

        (mtx, dist, _, _) =  self.__undistort_data
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
        return undistorted_img

