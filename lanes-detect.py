
import argparse
import logging
import configparser
from pipeline import Pipeline
from calibrate import Calibrate
from image_processing import ImageProcessing, ImageThresholding, ImageCannyEdgeDetection
from perspective import PerspectiveTransform

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Road Lanes detection')

parser.add_argument('-vidfile', help='video file to process')
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

    if args.cmd == "selfdiag":
        cam_calib = Calibrate(config)
        cam_calib.selfdiag()
        #pipeline = Pipeline(config)
        #pipeline.selfdiag()

        ip = ImageProcessing(config)
        ip.selfdiag()

        ip = ImageCannyEdgeDetection(config)
        ip.selfdiag()

        ip = ImageThresholding(config)
        ip.selfdiag()

        pt = PerspectiveTransform(config, load_saved_trans_matrix=False)
        pt.selfdiag()

        pass

    elif args.cmd == "calib":
        cam_calib = Calibrate(config)
        cam_calib.calibrate()

    elif args.cmd == "detect":
        # TODO
        pass

run()