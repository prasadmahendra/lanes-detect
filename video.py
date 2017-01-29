import logging
from moviepy.editor import VideoFileClip
from calibrate import Calibrate
from detection_pipeline import Pipeline

class VideoRender(object):
    __logger = logging.getLogger(__name__)

    def __init__(self, config, filename):
        self.__filename = filename
        self.__filename_out = filename + ".processed.mp4"
        self.__cam_calibration = Calibrate(config)
        self.__detection_pipe = Pipeline(config)

    def play(self):
        clip = VideoFileClip(self.__filename)
        clip = clip.fl_image(self.process_image)

        self.__logger.info("fps: {0}".format(clip.fps))
        self.__detection_pipe.set_fps(clip.fps)
        clip.write_videofile(self.__filename_out, audio=False)

    def process_image(self, image):
        return self.__detection_pipe.process(image)
