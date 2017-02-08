import logging
import cv2
from moviepy.editor import VideoFileClip
from calibrate import Calibrate
from detection_pipeline import Pipeline
from vehicle_search import VehicleSearch

class VideoRender(object):
    __logger = logging.getLogger(__name__)

    def __init__(self, config, filename):
        self.__config = config
        self.__filename = filename
        self.__filename_out_lanes_detect = filename + ".lanes.processed.mp4"
        self.__filename_out_vehicle_search = filename + ".vehicles.processed.mp4"
        self.__cam_calibration = Calibrate(config)
        self.__detection_pipe = Pipeline(config)
        self.__vehicle_search = None

    def play_lanes_detect(self):
        clip = VideoFileClip(self.__filename)
        clip = clip.fl_image(self.process_image_lanes_detect)

        self.__logger.info("fps: {0}".format(clip.fps))
        self.__detection_pipe.set_fps(clip.fps)
        clip.write_videofile(self.__filename_out_lanes_detect, audio=False)

    def play_vehicle_search(self):
        clip = VideoFileClip(self.__filename)

        self.__logger.info("fps: {0}".format(clip.fps))
        self.__vehicle_search = VehicleSearch(self.__config, clip.fps)

        clip = clip.fl_image(self.process_image_vehicle_search)
        clip.write_videofile(self.__filename_out_vehicle_search, audio=False)

    def process_image_lanes_detect(self, image):
        return cv2.cvtColor(self.__detection_pipe.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)

    def process_image_vehicle_search(self, image):
        return cv2.cvtColor(self.__vehicle_search.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)
