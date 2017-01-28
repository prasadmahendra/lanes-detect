import logging
import os
import cv2
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import uuid

from line import Line, LinesCollection
from calibrate import Calibrate
from image_processing import ImageProcessing, ImageThresholding, ImageCannyEdgeDetection
from perspective import PerspectiveTransform

class PipelineImageCollection(object):
    __logger = logging.getLogger(__name__)

    def __init__(self, image, image_undistort, thresholded_image, persp_image):
        self.orig_image = image
        self.image_undistort = image_undistort
        self.thresholded_image = thresholded_image
        self.persp_image = persp_image

class Pipeline(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        super(Pipeline, self).__init__(config)
        self.__config = config
        self.__cam_calibrate = Calibrate(config)
        self.__thresholding = ImageThresholding(config)
        self.__output_directory = config.get('detection', 'output_directory')
        self.__perspective_tran = PerspectiveTransform(config, load_saved_trans_matrix=True)
        self.__frame_no = 0
        self.__lines_collection = LinesCollection()

    def selfdiag(self):
        for filename, filepath in self.test_files():
            #self.process(self.load_image('data/test_images/test1.jpg'))
            self.process(self.load_image(filepath))
            #break

    def process(self, image, filename=None):
        self.__frame_no = self.__frame_no + 1

        image_undistort = self.undistort(image)
        thresholded_image = self.color_grad_threshold(image_undistort)
        persp_image = self.perspective_transform(thresholded_image)

        image_col = PipelineImageCollection(image, image_undistort, thresholded_image, persp_image)

        l_lane, r_lane = Line.to_lane_lines(self.__config, image_col, self.__frame_no)
        self.__lines_collection.append(l_lane, r_lane)

        persp_image_plot, img_with_lines = self.draw_lane_lines(image, persp_image)

        self.display_image_grid('detection-pipeline.jpg', [image, image_undistort, thresholded_image, persp_image, persp_image_plot, img_with_lines], ['image', 'image_undistort', 'thresholded', 'perspective', 'persp_image_plot', 'img_with_lines'], cmap='gray', save=self.save_output_images())
        return img_with_lines

    def undistort(self, image):
        return self.__cam_calibrate.undistort(image)

    def color_grad_threshold(self, image):
        return self.__thresholding.process(image, display=False)

    def perspective_transform(self, image):
        return self.__perspective_tran.process(image, display=False)

    def draw_lane_lines(self, undist_image, warped):
        l_lane, r_lane = self.__lines_collection.current()

        left_xvals = l_lane.xvals_polyfit()
        left_yvals = l_lane.yvals_polyfit()
        right_xvals = r_lane.xvals_polyfit()
        right_yvals = r_lane.yvals_polyfit()
        persp_image_plot = r_lane.line_fit_image()

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_xvals, left_yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xvals, right_yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        #if r_lane.is_generated():
        #    print("here! 1")
        #    cv2.polylines(color_warp, np.int_(pts_right), 10, (255, 0, 0))
        #if l_lane.is_generated():
        #    print("here! 2")
        #    cv2.polylines(color_warp, np.int_(pts_left), 10, (255, 0, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.__perspective_tran.get_minv(), (undist_image.shape[1], undist_image.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, "Curve L: {0}".format(int(l_lane.curve_rad())), (25, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "Curve R: {0}".format(int(r_lane.curve_rad())), (25, 150), font, 1, (255, 255, 255), 2)

        cv2.putText(result, "L: {0}".format(not l_lane.is_generated()), (300, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "R: {0}".format(not r_lane.is_generated()), (300, 150), font, 1, (255, 255, 255), 2)

        return persp_image_plot, result
