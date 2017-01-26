import logging
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from calibrate import Calibrate
from detection_pipeline import Pipeline

class VideoRender(object):
    def __init__(self, config, filename):
        self.__filename = filename
        self.__filename_out = filename + ".processed.mp4"
        self.__cam_calibration = Calibrate(config)
        self.__detection_pipe = Pipeline(config)

    def play(self):
        clip = VideoFileClip(self.__filename)
        clip = clip.fl_image(self.process_image)
        clip.write_videofile(self.__filename_out, audio=False)

    def process_image(self, image):
        return self.__detection_pipe.process(image)
