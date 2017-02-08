import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
import uuid
from collections import deque
from image_processing import ImageProcessing

class LinesCollection(object):
    __logger = logging.getLogger(__name__)
    #__logger.setLevel(logging.DEBUG)

    def __init__(self, video_fps):
        self.__left_lane_lines = deque([], maxlen=video_fps)   # keep the last video_fps lines
        self.__right_lane_lines = deque([], maxlen=video_fps)  # keep the last video_fps lines
        self.__processed_frames_total = 0
        self.__detected_left_lanes_count = 0
        self.__detected_right_lanes_count = 0
        self.__dropped_left_lanes_count = 0
        self.__dropped_right_lanes_count = 0
        self.__curr_vehicle_position = 0
        self.__line_averages_over = video_fps

    def set_line_averages_over(self, line_averages_over):
        self.__line_averages_over = line_averages_over

    def dropped_frames_count(self):
        return self.__dropped_left_lanes_count, self.__dropped_right_lanes_count

    def total_frames_processed(self):
        return self.__processed_frames_total

    def detected_lanes_count(self):
        return self.__detected_left_lanes_count, self.__detected_right_lanes_count

    def append(self, left_line, right_line):
        self.__processed_frames_total += 1

        assert(left_line.rol() == 'left')
        assert(right_line.rol() == 'right')

        if not left_line.is_detected():
            left_line = self.find_best_fit(left_line, self.__left_lane_lines, fallback=right_line)
        else:
            self.__detected_left_lanes_count += 1

        if not right_line.is_detected():
            right_line = self.find_best_fit(right_line, self.__right_lane_lines, fallback=left_line)
        else:
            self.__detected_right_lanes_count += 1

        assert(left_line.rol() == 'left')
        assert(right_line.rol() == 'right')

        left_line, right_line = self.smooth_lines(left_line, right_line)

        assert(left_line.rol() == 'left')
        assert(right_line.rol() == 'right')

        self.__curr_vehicle_position = self.vehicle_position(left_line, right_line)

        self.__left_lane_lines.appendleft(left_line)
        self.__right_lane_lines.appendleft(right_line)

    def current_vehicle_position(self):
        return self.__curr_vehicle_position

    def vehicle_position(self, left_line, right_line):
        xvals_l = left_line.xvals_polyfitx()
        xvals_r = right_line.xvals_polyfitx()

        lane_center_x = xvals_l[0] + (abs(xvals_l[0] - xvals_r[0]) / 2)
        vehicle_center_x = left_line.image_collection().orig_image.shape[1] / 2

        diff = (vehicle_center_x - lane_center_x) * left_line.meters_per_pixel_x()
        return diff;

    def lane_width_diff_correction(self, line_l, prev_line_l, line_r, prev_line_r):
        if prev_line_l is None or prev_line_r is None:
            return line_l, line_r
        else:
            xvals_l = line_l.xvals_polyfitx()
            xvals_r = line_r.xvals_polyfitx()
            prev_xvals_l = prev_line_l.xvals_polyfitx()
            prev_xvals_r = prev_line_r.xvals_polyfitx()

            detected_lane_width = abs(xvals_l[0] - xvals_r[0]) * line_l.meters_per_pixel_x()
            prev_detected_lane_width = abs(prev_xvals_l[0] - prev_xvals_r[0]) * prev_line_l.meters_per_pixel_x()

            diff = (abs(detected_lane_width - prev_detected_lane_width) / max(detected_lane_width, prev_detected_lane_width)) * 100
            if diff > 10.0:
                self.__logger.warning("Discard bad lane lines. Curr lane width: {0} prev lane width: {1} diff: {2}".format(detected_lane_width, prev_detected_lane_width, diff))
                self.__dropped_left_lanes_count += 1
                self.__dropped_right_lanes_count += 1
                return prev_line_l, prev_line_r
            else:
                self.__logger.debug("xvals_l: %s xvals_r: %s prev xvals_l: %s prev xvals_r: %s", xvals_l[0], xvals_r[0], prev_xvals_l[0], prev_xvals_r[0])
                self.__logger.debug("curr lane width: {0} prev lane width: {1} diff: {2}".format(detected_lane_width, prev_detected_lane_width, diff))
                return line_l, line_r


    def smooth_lines(self, left_line, right_line):
        prev_lane_l = None
        prev_lane_r = None

        if len(self.__left_lane_lines) > 0 and len(self.__right_lane_lines) > 0:
            prev_lane_l = self.__left_lane_lines[0]
            prev_lane_r = self.__right_lane_lines[0]

        left_line, right_line = self.lane_width_diff_correction(left_line, prev_lane_l, right_line, prev_lane_r)

        curverad_diff_perc = self.curve_rad_diff(left_line, prev_lane_l)
        if  curverad_diff_perc > 1000:
            # discard ....
            self.__logger.warning("Line {0} dropped. curverad_diff_perc: {1} ({2} - {3})".format(left_line.rol(), curverad_diff_perc, left_line.curve_rad(), prev_lane_l.curve_rad()))
            self.__dropped_left_lanes_count += 1
            left_line = prev_lane_l

        curverad_diff_perc = self.curve_rad_diff(right_line, prev_lane_r)
        if  curverad_diff_perc > 1000:
            # discard ....
            self.__logger.warning("Line {0} dropped. curverad_diff_perc: {1} ({2} - {3})".format(right_line.rol(), curverad_diff_perc, right_line.curve_rad(), prev_lane_r.curve_rad()))
            self.__dropped_right_lanes_count += 1
            right_line = prev_lane_r

        polyfit_l = np.copy(left_line.polyfit())
        polyfit_r = np.copy(right_line.polyfit())
        yvals_l = np.copy(left_line.yvals_polyfitx())
        yvals_r = np.copy(right_line.yvals_polyfitx())
        xvals_l = np.copy(left_line.xvals_polyfitx())
        xvals_r = np.copy(right_line.xvals_polyfitx())

        prev_frame = 0
        samples = 1
        while prev_frame < len(self.__left_lane_lines) and prev_frame < len(self.__right_lane_lines) and prev_frame < self.__line_averages_over:
            prev_lane_l = self.__left_lane_lines[prev_frame]
            prev_lane_r = self.__right_lane_lines[prev_frame]

            polyfit_l += np.copy(prev_lane_l.polyfit())
            polyfit_r += np.copy(prev_lane_r.polyfit())
            yvals_l += np.copy(left_line.yvals_polyfitx())
            yvals_r += np.copy(right_line.yvals_polyfitx())
            xvals_l += np.copy(left_line.xvals_polyfitx())
            xvals_r += np.copy(right_line.xvals_polyfitx())

            prev_frame += 1
            samples += 1

        polyfit_l = polyfit_l / samples
        polyfit_r = polyfit_r / samples
        yvals_l = yvals_l / samples
        yvals_r = yvals_r / samples
        xvals_l = xvals_l / samples
        xvals_r = xvals_r / samples

        #xvals_polyfitx_l = polyfit_l[0] * left_line.yvals_polyfitx() ** 2 + polyfit_l[1] * left_line.yvals_polyfitx() + polyfit_l[2]
        #left_line.set_best_polyfit_vals(polyfit_l, xvals_polyfitx_l, left_line.yvals_polyfitx())
        left_line.set_best_polyfit_vals(polyfit_l, xvals_l, yvals_l)

        #xvals_polyfitx_r = polyfit_r[0] * right_line.yvals_polyfitx() ** 2 + polyfit_r[1] * right_line.yvals_polyfitx() + polyfit_r[2]
        #right_line.set_best_polyfit_vals(polyfit_r, xvals_polyfitx_r, right_line.yvals_polyfitx())
        right_line.set_best_polyfit_vals(polyfit_r, xvals_r, yvals_r)

        return left_line, right_line

    def curve_rad_diff(self, line, prev_line):
        if prev_line is None:
            return 0

        diff = (abs(line.curve_rad() - prev_line.curve_rad()) / min(line.curve_rad(), prev_line.curve_rad())) * 100
        if diff > 1000:
            self.__logger.debug("{0} curvature diff: {1} ({2} - {3} / {4})".format(line.rol(), diff, line.curve_rad(), prev_line.curve_rad(), min(line.curve_rad(), prev_line.curve_rad())))
        else:
            self.__logger.debug("{0} curvature diff: {1} ({2} - {3} / {4})".format(line.rol(), diff, line.curve_rad(),
                                                                                   prev_line.curve_rad(),
                                                                                   min(line.curve_rad(),
                                                                                       prev_line.curve_rad())))
        return diff

    def find_best_fit(self, line, lane_lines_queue, fallback):
        best_fit_line = None
        if len(lane_lines_queue) > 0:
            best_fit_line = lane_lines_queue[0] # get the last line ...
            self.__logger.info("Line ({0}) detection failed. Pick prev line ({1})".format(line.rol(), best_fit_line.rol()))
        else:
            if fallback.is_detected():
                if fallback.rol() == 'left':
                    self.__logger.info("Line ({0}) detection failed. Pick fallback line ({1})".format(line.rol(), fallback.rol()))
                    best_fit_line = fallback.right_shift()
                else:
                    self.__logger.info("Line ({0}) detection failed. Pick fallback line ({1})".format(line.rol(), fallback.rol()))
                    best_fit_line = fallback.left_shift()
            else:
                self.__logger.info("Best fit for %s line (frame %s) not possible. lane lines q: %s fallback: %s", line.rol(), line.frame_no(), len(lane_lines_queue), fallback.is_detected())
                raise Exception("Best fit failed")

        assert(best_fit_line is not None)

        best_fit_line.set_generated(True)
        self.__dropped_right_lanes_count += 1

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
        self.__xm_per_pix = self.lane_width() / image_width  # meters per pixel in x dimension

        self.__xvals = None
        self.__yvals = None

        self.__curverad = None
        self.__xvals_polyfit = None
        self.__xvals_polyfitx = None
        self.__yvals_polyfitx = None
        self.__xvals_polyfit_best = None
        self.__xvals_polyfitx_best = None
        self.__yvals_polyfitx_best= None
        self.__line_fit_image = None

        self.__line_color = 'green'
        if self.rol() == 'left':
            self.__xvals_color = 'red'
        else:
            self.__xvals_color = 'blue'

        self.__generated = False

    def copy(self):
        config = self.__config
        self.__config = None
        cpy = copy.deepcopy(self)
        cpy.__config = config
        return cpy

    def set_generated(self, gen):
        self.__generated = gen

    def is_generated(self):
        return self.__generated

    def set_best_polyfit_vals(self, polyfit_best, xvals_polyfitx_best, yvals_polyfitx_best):
        assert (polyfit_best is not None)
        assert (xvals_polyfitx_best is not None)
        assert (yvals_polyfitx_best is not None)

        self.__xvals_polyfit_best = polyfit_best
        self.__xvals_polyfitx_best = xvals_polyfitx_best
        self.__yvals_polyfitx_best = yvals_polyfitx_best

    def polyfit_coeffs_best(self):
        return self.__xvals_polyfit_best

    def xvals_polyfitx_best(self):
        return self.__xvals_polyfitx_best

    def yvals_polyfitx_best(self):
        return self.__yvals_polyfitx_best

    def polyfit_coeffs(self):
        return self.__xvals_polyfit

    def xvals_polyfitx(self):
        return self.__xvals_polyfitx

    def yvals_polyfitx(self):
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
            self.__xval_start = np.argmax(self.__histogram[0:int(len(self.__histogram) / 2)])
        else:
            self.__xval_start = np.argmax(self.__histogram[int(len(self.__histogram) / 2):len(self.__histogram)]) + int(len(self.__histogram)/2)

        self.__yvals, self.__xvals = self.to_lane_pixels(self.__image_col, self.__xval_start)

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

        yvals = np.append(np.asarray([image_height]), self.__yvals).flatten()
        xvals = np.append(np.asarray([self.__xval_start]), self.__xvals).flatten()

        xvals_polyfit = np.polyfit(yvals, xvals, 2)
        xvals_polyfitx = xvals_polyfit[0] * yvals_extended ** 2 + xvals_polyfit[1] * yvals_extended + xvals_polyfit[2]

        # Plot the left and right lane pixes ...
        plt.plot(self.__xvals, self.__yvals, 'o', color=self.__xvals_color)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)

        # Plot the second order polynomial for lane lines ...
        plt.plot(xvals_polyfitx, yvals_extended, color=self.__line_color, linewidth=3)
        plt.gca().invert_yaxis()  # to visualize as we do the images

        self.__curverad = self.determine_curvature(persp_image, xvals_polyfitx, yvals_extended)
        self.__xvals_polyfit = xvals_polyfit
        self.__xvals_polyfitx = xvals_polyfitx
        self.__yvals_polyfitx = yvals_extended

        self.__detected = True

        self.__line_fit_image = "{0}/tmp/frame_{1}.png".format(self.__output_directory, self.frame_no())
        plt.plot(self.__histogram)
        plt.savefig(self.__line_fit_image)

        return

    def polyfit(self):
        return self.__xvals_polyfit

    def curve_rad(self):
        return self.__curverad

    def determine_curvature(self, image, xvals, yvals):
        # Define y-value where we want radius of curvature. Choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(yvals)

        fit_cr = np.polyfit(yvals * self.__ym_per_pix, xvals * self.__xm_per_pix, 2)

        # take the curvature of the line at the midpoint (y_eval / 2.)
        curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) \
                   / np.absolute(2 * fit_cr[0])

        return curverad

    def to_lane_pixels(self, image_col, xval_start):
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
        new_line = self.copy()
        new_line.__xvals = new_line.__xvals + (self.meters_per_pixel_x() * self.lane_width())
        new_line.__rol = 'right'
        return new_line

    def left_shift(self):
        self.__logger.info("Swap lines L <- R. Frame: %s", self.__frame_no)
        new_line = self.copy()
        new_line.__xvals = new_line.__xvals - (self.meters_per_pixel_x() * self.lane_width())
        new_line.__rol = 'left'
        return new_line

    def meters_per_pixel_x(self):
        return self.__xm_per_pix

    def lane_width(self):
        return 3.7

    def image_collection(self):
        return self.__image_col

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