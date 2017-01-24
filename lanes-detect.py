
import argparse
import logging
import configparser
from calibrate import Calibrate

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

        pass

    elif args.cmd == "calib":
        # TODO
        pass

    elif args.cmd == "detect":
        # TODO
        pass

run()