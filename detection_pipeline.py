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

from calibrate import Calibrate
from image_processing import ImageProcessing, ImageThresholding, ImageCannyEdgeDetection
from perspective import PerspectiveTransform

class Pipeline(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        super(Pipeline, self).__init__(config)
        self.__cam_calibrate = Calibrate(config)
        self.__thresholding = ImageThresholding(config)
        self.__perspective_tran = PerspectiveTransform(config, load_saved_trans_matrix=True)

    def selfdiag(self):
        for filename, filepath in self.test_files():
            #self.process(self.load_image('data/test_images/test1.jpg'))
            self.process(self.load_image(filepath))
            #break

    def process(self, image, filename=None):
        image_undistort = self.undistort(image)
        thresholded = self.color_grad_threshold(image_undistort)
        persp_tran = self.perspective_transform(thresholded)

        lane_lines_img, lane_lines_img, left_xvals_fitx, left_yvals, right_xvals_fitx, right_yvals = self.detect_lane_lines(persp_tran)

        img_with_lines = self.draw_lane_lines(image, persp_tran, left_xvals_fitx, left_yvals, right_xvals_fitx, right_yvals)

        self.display_image_grid('detection-pipeline.jpg', [image, image_undistort, thresholded, persp_tran, lane_lines_img, img_with_lines], ['image', 'image_undistort', 'thresholded', 'perspective', 'lane_lines_img', 'img_with_lines'], cmap='gray', save=self.save_output_images())
        return image

    def undistort(self, image):
        return self.__cam_calibrate.undistort(image)

    def color_grad_threshold(self, image):
        return self.__thresholding.process(image, display=False)

    def perspective_transform(self, image):
        return self.__perspective_tran.process(image, display=False)

    def detect_lane_lines(self, image):
        histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)
        plt.plot(histogram)

        left_line_x = np.argmax(histogram[0:int(len(histogram) / 2)])
        right_line_x = np.argmax(histogram[int(len(histogram) / 2):len(histogram)]) + int(len(histogram)/2)

        left_lane_yvals, left_lane_xvals = self.to_lane_pixes(image, left_line_x, orient='left')
        right_lane_yvals, right_lane_xvals = self.to_lane_pixes(image, right_line_x, orient='right')

        lane_lines_img, left_xvals_fitx, left_yvals_extended, right_xvals_fitx, right_yvals_extended = self.line_fit(image, left_lane_yvals, left_lane_xvals, right_lane_yvals, right_lane_xvals)

        return lane_lines_img, lane_lines_img, left_xvals_fitx, left_yvals_extended, right_xvals_fitx, right_yvals_extended

    def to_lane_pixes(self, image, xval_start, orient='left'):
        y_max = image.shape[0] - self.engine_compart_pixes()
        y_min = max(0, y_max - 25)

        x_max = min(xval_start + 75, image.shape[1])
        x_min = max(0, xval_start - 75)

        yvals = None
        xvals = None

        xval_prev = xval_start
        xval = xval_start

        while y_max > 0:
            search_bounding_box = image[y_min:y_max, x_min:x_max]
            x_y_indices = np.where(search_bounding_box == 1)

            if len(x_y_indices[0]) > 0:
                if xvals is None:
                    xvals = x_y_indices[0] + x_min
                    yvals = x_y_indices[1] + y_max - self.engine_compart_pixes()
                else:
                    xvals = np.append(xvals, x_y_indices[0] + x_min)
                    yvals = np.append(yvals, x_y_indices[1] + y_max - self.engine_compart_pixes())

            #im = search_bounding_box
            #if img is None:
            #    img = plt.imshow(im, cmap='gray')
            #else:
            #    img.set_data(im)
            #plt.pause(.1)
            #plt.draw()

            histogram = np.sum(search_bounding_box[int(search_bounding_box.shape[0] / 2):, :], axis=0)
            adjusted_peak = np.argmax(histogram)

            if adjusted_peak < 10:
                #xval = xval_start
                xval = xval_prev
            else:
                xval_prev = xval
                xval = x_min + adjusted_peak

            x_max = min(xval + 75, image.shape[1])
            x_min = max(0, xval - 75)

            y_max = y_max - 25
            y_min = max(0, y_max - 25)

        xvals = xvals.flatten()
        yvals = yvals.flatten()

        return yvals, xvals


    def line_fit(self, image, left_yvals, left_xvals, right_yvals, right_xvals, left_xvals_color='red', left_line_color='green', right_xvals_color='blue', right_line_color='green'):
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Fit a second order polynomial to each lane line ...
        left_xvals_fit = np.polyfit(left_yvals, left_xvals, 2)
        left_yvals_extended = np.append(left_yvals, np.asarray([0])).flatten()
        left_yvals_extended = np.append(np.asarray([image_height]), left_yvals_extended).flatten()
        left_xvals_fitx = left_xvals_fit[0] * left_yvals_extended ** 2 + left_xvals_fit[1] * left_yvals_extended + left_xvals_fit[2]

        right_xvals_fit = np.polyfit(right_yvals, right_xvals, 2)
        right_yvals_extended = np.append(right_yvals, np.asarray([0])).flatten()
        right_yvals_extended = np.append(np.asarray([image_height]), right_yvals_extended).flatten()
        right_xvals_fitx = right_xvals_fit[0] * right_yvals_extended ** 2 + right_xvals_fit[1] * right_yvals_extended + right_xvals_fit[2]

        # Plot the left and right lane pixes ...
        plt.plot(left_xvals, left_yvals, 'o', color=left_xvals_color)
        plt.plot(right_xvals, right_yvals, 'o', color=right_xvals_color)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)

        # Plot the second order polynomial for lane lines ...
        plt.plot(left_xvals_fitx, left_yvals_extended, color=left_line_color, linewidth=3)
        plt.plot(right_xvals_fitx, right_yvals_extended, color=right_line_color, linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images

        temp_file = "/tmp/{0}.png".format(uuid.uuid1())
        plt.savefig(temp_file)
        img = self.load_image(temp_file)
        os.remove(temp_file)
        plt.close()

        (left_curverad, right_curverad) = self.determine_curvature(image, left_xvals, left_yvals, right_xvals, right_yvals)

        return img, left_xvals_fitx, left_yvals_extended, right_xvals_fitx, right_yvals_extended

    def determine_curvature(self, image, left_xvals, left_yvals, right_xvals, right_yvals):
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Define y-value where we want radius of curvature. Choose the maximum y-value, corresponding to the bottom of the image
        left_y_eval = np.max(left_yvals)
        right_y_eval = np.max(right_yvals)

        ym_per_pix = 30 / image_height  # meters per pixel in y dimension
        xm_per_pix = 3.7 / image_width  # meteres per pixel in x dimension

        left_fit_cr = np.polyfit(left_yvals * ym_per_pix, left_xvals * xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_yvals * ym_per_pix, right_xvals * xm_per_pix, 2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * left_y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * right_y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def draw_lane_lines(self, undist_image, warped, left_xvals, left_yvals, right_xvals, right_yvals):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_xvals, left_yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_xvals, right_yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.__perspective_tran.get_minv(), (undist_image.shape[1], undist_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

        return result
