
import argparse
import logging
import configparser
from detection_pipeline import Pipeline
from calibrate import Calibrate
from image_processing import ImageProcessing, ImageThresholding, ImageCannyEdgeDetection
from perspective import PerspectiveTransform
from video import VideoRender

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Road Lanes detection')

parser.add_argument('-video', default='data/test_videos/project_video.mp4', help='video file to process')
parser.add_argument('-cmd', default='selfdiag', help='Commands (default: selfdiag)', choices=['selfdiag', 'calib', 'detect'])
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")

args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

def run():
    config = configparser.RawConfigParser()
    config.read("settings.ini")
    logger.info("Running cmd: %s" % (args.cmd))

    args.cmd = 'detect'

    if args.cmd == "selfdiag":
        cam_calib = Calibrate(config)
        cam_calib.selfdiag()

        ip = ImageProcessing(config)
        ip.selfdiag()

        ip = ImageCannyEdgeDetection(config)
        ip.selfdiag()

        ip = ImageThresholding(config)
        ip.selfdiag()

        pt = PerspectiveTransform(config, load_saved_trans_matrix=False)
        pt.selfdiag()

        pipeline = Pipeline(config)
        pipeline.selfdiag()

    elif args.cmd == "calib":
        cam_calib = Calibrate(config)
        cam_calib.calibrate()

    elif args.cmd == "detect":
        vid = VideoRender(config, args.video)
        vid.play()

run()