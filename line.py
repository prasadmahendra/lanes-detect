import logging
import copy
import os
import cv2
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import uuid
from collections import deque
from image_processing import ImageProcessing

class LinesCollection(object):
    __logger = logging.getLogger(__name__)

    def __init__(self):
        self.__left_lane_lines = deque([], maxlen=10)        # keep the last 10 lines
        self.__right_lane_lines = deque([], maxlen=10)  # keep the last 10 lines
        pass

    def append(self, left_line, right_line):
        # add a new entry to the right side
        if left_line.is_detected():
            self.__left_lane_lines.appendleft(left_line)
        else:
            self.__logger.info("Frame %s left line not detected. Find best fit ...", left_line.frame_no())
            left_best_fit = self.find_best_fit(left_line, self.__left_lane_lines, fallback=right_line)
            left_best_fit.set_generated(True)
            self.__left_lane_lines.appendleft(left_best_fit)

        if right_line.is_detected():
            self.__right_lane_lines.appendleft(right_line)
        else:
            self.__logger.info("Frame %s right line not detected. Find best fit ...", right_line.frame_no())
            right_best_fit = self.find_best_fit(right_line, self.__right_lane_lines, fallback=left_line)
            right_best_fit.set_generated(True)
            self.__right_lane_lines.appendleft(right_best_fit)

    def find_best_fit(self, line, lane_lines_queue, fallback):
        best_fit_line = None
        if len(lane_lines_queue) > 0:
            best_fit_line = lane_lines_queue[0] # get the last line ...
        else:
            if fallback.is_detected():
                if fallback.rol() == 'left':
                    best_fit_line = fallback.right_shift()
                else:
                    best_fit_line = fallback.left_shift()
            else:
                self.__logger.info("Best fit for %s line (frame %s) not possible. lane lines q: %s fallback: %s", line.lor(), line.frame_no(), len(lane_lines_queue), fallback.is_detected())
                raise Exception("Best fit failed")

        assert(best_fit_line is not None)
        return best_fit_line

    def current(self):
        if len(self.__left_lane_lines) > 0:
            # peek at leftmost item
            return self.__left_lane_lines[0], self.__right_lane_lines[0]
        else:
            return None, None


class Line(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config, image_col, frame_no, rol='left|right'):
        self.__config = config
        self.__output_directory = config.get('detection', 'output_directory')
        self.__image_col = image_col
        self.__persp_image = image_col.persp_image
        image_height = self.__persp_image.shape[0]
        image_width = self.__persp_image.shape[1]

        self.__detected = False
        self.__frame_no = frame_no
        self.__rol = rol
        self.__ym_per_pix = 30 / image_height  # meters per pixel in y dimension
        self.__xm_per_pix = 3.7 / image_width  # meters per pixel in x dimension

        self.__xvals = None
        self.__yvals = None

        self.__curverad = None
        self.__xvals_polyfit = None
        self.__xvals_polyfitx = None
        self.__yvals_polyfitx = None
        self.__line_fit_image = None

        self.__line_color = 'green'
        if self.rol() == 'left':
            self.__xvals_color = 'red'
        else:
            self.__xvals_color = 'blue'

        self.__generated = False

    def set_generated(self, gen):
        self.__generated = gen

    def is_generated(self):
        return self.__generated

    def xvals_polyfit(self):
        return self.__xvals_polyfitx

    def yvals_polyfit(self):
        return self.__yvals_polyfitx

    def to_lane_lines(config, image_col, frame_no):
        Line.__logger.debug("Extract left lane line. frame %s ....", frame_no)

        left_l = Line(config, image_col, frame_no, rol='left')
        left_l.extract_lane_pixels()
        left_l.line_fit()

        Line.__logger.debug("Extract right lane line. frame %s ....", frame_no)

        right_l = Line(config, image_col, frame_no, rol='right')
        right_l.extract_lane_pixels()
        right_l.line_fit()

        plt.close()

        return left_l, right_l

    def extract_lane_pixels(self):
        self.__histogram = np.sum(self.__image_col.persp_image[int(self.__image_col.persp_image.shape[0] / 2):, :], axis=0)

        if self.rol() == 'left':
            xval_start = np.argmax(self.__histogram[0:int(len(self.__histogram) / 2)])
        else:
            xval_start = np.argmax(self.__histogram[int(len(self.__histogram) / 2):len(self.__histogram)]) + int(len(self.__histogram)/2)

        self.__yvals, self.__xvals = self.to_lane_pixes(self.__image_col, xval_start)

    def line_fit(self):
        if self.__xvals is None or self.__yvals is None:
            self.__detected = False
            return

        persp_image = self.__persp_image
        image_height = persp_image.shape[0]
        image_width = persp_image.shape[1]

        # Fit a second order polynomial to each lane line ...
        # force extend yvals to top
        yvals_extended = np.append(self.__yvals, np.asarray([0])).flatten()
        # force extend yvals to bottom
        yvals_extended = np.append(np.asarray([image_height]), yvals_extended).flatten()

        xvals_polyfit = np.polyfit(self.__yvals, self.__xvals, 2)
        xvals_polyfitx = xvals_polyfit[0] * yvals_extended ** 2 + xvals_polyfit[1] * yvals_extended + xvals_polyfit[2]

        # Plot the left and right lane pixes ...
        plt.plot(self.__xvals, self.__yvals, 'o', color=self.__xvals_color)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)

        # Plot the second order polynomial for lane lines ...
        plt.plot(xvals_polyfitx, yvals_extended, color=self.__line_color, linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images

        self.__curverad = self.determine_curvature(persp_image, self.__xvals, self.__yvals)
        self.__xvals_polyfit = xvals_polyfit
        self.__xvals_polyfitx = xvals_polyfitx
        self.__yvals_polyfitx = yvals_extended

        self.__detected = True

        self.__line_fit_image = "{0}/tmp/frame_{1}.png".format(self.__output_directory, self.frame_no())
        plt.plot(self.__histogram)
        plt.savefig(self.__line_fit_image)

        return

    def curve_rad(self):
        return self.__curverad

    def determine_curvature(self, image, xvals, yvals):
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Define y-value where we want radius of curvature. Choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(yvals)

        ym_per_pix = 30 / image_height  # meters per pixel in y dimension
        xm_per_pix = 3.7 / image_width  # meteres per pixel in x dimension

        fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)

        curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * fit_cr[0])

        return curverad

    def to_lane_pixes(self, image_col, xval_start):
        y_max = image_col.persp_image.shape[0] - self.engine_compart_pixes()
        y_min = max(0, y_max - 25)

        x_max = min(xval_start + 75, image_col.persp_image.shape[1])
        x_min = max(0, xval_start - 75)

        yvals = None
        xvals = None

        xval_prev = xval_start
        xval = xval_start

        while y_max > 0:
            search_bounding_box = image_col.persp_image[y_min:y_max, x_min:x_max]
            x_y_indices = np.where(search_bounding_box == 1)

            if len(x_y_indices[0]) > 0:
                if xvals is None:
                    xvals = x_y_indices[1] + x_min
                    yvals = x_y_indices[0] + y_max - self.engine_compart_pixes()
                else:
                    xvals = np.append(xvals, x_y_indices[1] + x_min)
                    yvals = np.append(yvals, x_y_indices[0] + y_max - self.engine_compart_pixes())

            histogram = np.sum(search_bounding_box[int(search_bounding_box.shape[0] / 2):, :], axis=0)
            adjusted_peak = np.argmax(histogram)

            if adjusted_peak < 10:
                #xval = xval_start
                xval = xval_prev
            else:
                xval_prev = xval
                xval = x_min + adjusted_peak

            x_max = min(xval + 75, image_col.persp_image.shape[1])
            x_min = max(0, xval - 75)

            y_max = y_max - 25
            y_min = max(0, y_max - 25)

        if xvals is not None:
            xvals = xvals.flatten()
        else:
            self.__logger.info("line (%s) detection failed (x) frame_no: %s", self.rol(), self.__frame_no)
            self.__save_image(image_col)

        if yvals is not None:
            yvals = yvals.flatten()
        else:
            self.__logger.info("line (%s) detection failed (y) frame no: %s", self.rol(), self.__frame_no)
            self.__save_image(image_col)

        return yvals, xvals

    def line_fit_image(self):
        file = "{0}/tmp/frame_{1}.png".format(self.__output_directory, self.frame_no())
        self.__logger.debug("loading %s", file)
        img = self.load_image(file)

        #if delete and os.path.isfile(file):
        #    os.remove(file)

        return img

    def is_detected(self):
        return self.__detected

    def frame_no(self):
        return self.__frame_no

    def rol(self):
        return self.__rol

    def right_shift(self):
        self.__logger.info("Swap lines L -> R. Frame: %s", self.__frame_no)
        new_line = copy.deepcopy(self)
        new_line.__xvals = new_line.__xvals + (self.meters_per_pixel_x() * self.lane_width())
        return new_line

    def left_shift(self):
        self.__logger.info("Swap lines L <- R. Frame: %s", self.__frame_no)
        new_line = copy.deepcopy(self)
        new_line.__xvals = new_line.__xvals - (self.meters_per_pixel_x() * self.lane_width())
        return new_line

    def meters_per_pixel_x(self):
        return self.__xm_per_pix

    def lane_width(self):
        return 3.7

    def __save_image(self, image_col):
        fig, axn = plt.subplots(1, 3, figsize=(24, 9))
        fig.tight_layout()

        image_undistort = image_col.image_undistort
        thresholded_image = image_col.thresholded_image
        persp_image = image_col.persp_image

        axn[0].imshow(image_undistort, cmap='gray')
        axn[0].set_title('image_undistort', fontsize=40)
        axn[1].imshow(thresholded_image, cmap='gray')
        axn[1].set_title('thresholded_image', fontsize=40)
        axn[2].imshow(persp_image, cmap='gray')
        axn[2].set_title('persp_image', fontsize=40)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        filename = "{0}.jpg".format(uuid.uuid4())
        self.__logger.info("saving {0}/errors/{1}".format(self.__output_directory, filename))
        fig.savefig("{0}/errors/{1}".format(self.__output_directory, filename))

        plt.close()