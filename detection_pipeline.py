import logging
import cv2
import numpy as np

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
        self.__fps = 1
        self.__lines_collection = LinesCollection(video_fps=self.__fps)

    def selfdiag(self):
        for filename, filepath in self.test_files():
            self.__logger.info("process: {0}".format(filename))
            self.__fps = 1
            self.process(self.load_image(filepath))

    def set_fps(self, fps):
        self.__fps = fps
        self.__lines_collection.set_line_averages_over(self.__fps)

    def process(self, image, filename=None):
        self.__frame_no = self.__frame_no + 1

        image_undistort = self.undistort(image)
        thresholded_image = self.color_grad_threshold(image_undistort)
        image_roi = self.region_of_interest(image, binary_image=False)
        thresholded_image_roi = self.region_of_interest(thresholded_image, binary_image=True)
        persp_image = self.perspective_transform(thresholded_image_roi)

        image_col = PipelineImageCollection(image, image_undistort, thresholded_image, persp_image)

        l_lane, r_lane = Line.to_lane_lines(self.__config, image_col, self.__frame_no)
        self.__lines_collection.append(l_lane, r_lane)

        persp_image_plot, img_with_lines = self.draw_lane_lines(image, persp_image)

        self.display_image_grid('detection-pipeline', "frame-{0}.jpg".format(self.__frame_no), [image, image_roi, image_undistort, thresholded_image_roi, persp_image, persp_image_plot, img_with_lines], ['image', 'image_roi', 'image_undistort', 'thresholded_image_roi', 'perspective', 'persp_image_plot', 'img_with_lines'], cmap='gray', save=self.save_output_images())
        return img_with_lines

    def undistort(self, image):
        return self.__cam_calibrate.undistort(image)

    def color_grad_threshold(self, image):
        return self.__thresholding.process(image, display=False)

    def perspective_transform(self, image):
        return self.__perspective_tran.process(image, display=False)

    def draw_lane_lines(self, undist_image, warped):
        l_lane, r_lane = self.__lines_collection.current()
        l_lanes_dropped, r_lanes_dropped = self.__lines_collection.dropped_frames_count()
        l_lanes_detected, r_lanes_detected = self.__lines_collection.detected_lanes_count()
        total_processed = self.__lines_collection.total_frames_processed()
        vehicle_position = self.__lines_collection.current_vehicle_position()

        left_xvals = l_lane.xvals_polyfitx_best()
        left_yvals = l_lane.yvals_polyfitx_best()
        right_xvals = r_lane.xvals_polyfitx_best()
        right_yvals = r_lane.yvals_polyfitx_best()
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

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.__perspective_tran.get_minv(), (undist_image.shape[1], undist_image.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, "Curve L: %6sm" % int(l_lane.curve_rad()), (25, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "Curve R: %6sm" % int(r_lane.curve_rad()), (25, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "Center  : %6sm" % round(vehicle_position, 2), (25, 150), font, 1, (255, 255, 255), 2)

        cv2.putText(result, "L Detected: {0} / {1}".format(l_lanes_detected, total_processed), (350, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "R Detected: {0} / {1}".format(r_lanes_detected, total_processed), (350, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "L Dropped: {0}".format(l_lanes_dropped), (350, 150), font, 1, (255, 255, 255), 2)
        cv2.putText(result, "R Dropped: {0}".format(r_lanes_dropped), (350, 200), font, 1, (255, 255, 255), 2)

        return persp_image_plot, result


