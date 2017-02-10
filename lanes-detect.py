import cv2
import argparse
import logging
import configparser
from detection_pipeline import Pipeline
from calibrate import Calibrate
from image_processing import ImageProcessing, ImageThresholding, ImageCannyEdgeDetection
from vehicle_detection import VehicleDetection
from perspective import PerspectiveTransform
from video import VideoRender

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Road Lane lines & vehicles detection')

parser.add_argument('-video', default='data/test_videos/project_video.mp4', help='video file to process')
parser.add_argument('-cmd', default='detect-vehicles', help='Commands (default: selfdiag)', choices=['selfdiag', 'calib', 'detect', 'vehicle-detect-train', 'vehicle-detect-predict', 'detect-vehicles'])
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument('-imagefile', default='data/test_images/image0010.png', help='image file (path) to process')

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
        vid.play_lanes_detect()

    elif args.cmd == "detect-vehicles":
        vid = VideoRender(config, args.video)
        vid.play_vehicle_search()

    elif args.cmd == 'undistort':
        cam_calib = Calibrate(config)
        ip = ImageProcessing(config)
        image_orig = cv2.imread(args.imagefile)
        image_undistort = cam_calib.undistort(image_orig)

        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        image_undistort = cv2.cvtColor(image_undistort, cv2.COLOR_BGR2RGB)
        ip.display_image_grid('camera_cal', 'undistorted.jpg', [image_orig, image_undistort], ['orig', 'undistorted'], save=True)

    elif args.cmd == 'vehicle-detect-train':
        vdetect = VehicleDetection(config)
        vdetect.train()

    elif args.cmd == 'vehicle-detect-predict':
        ip = ImageProcessing(config)
        image = ip.load_image(args.imagefile)
        vdetect = VehicleDetection(config)
        p = vdetect.predict(image)
        if p == 1:
            print("It's a Car!")
        else:
            print("I see no Cars!")


run()