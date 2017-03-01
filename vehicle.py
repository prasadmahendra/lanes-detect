import logging
import copy
import numpy as np
import cv2
import uuid
from collections import deque
from image_processing import ImageProcessing

class VehiclesCollection(ImageProcessing):
    __logger = logging.getLogger(__name__)
    MIN_CENTROID_DISTANCE = 32
    MIN_CENTROIDS_THRESHOLD = 2

    def __init__(self, config, video_fps):
        super(VehiclesCollection, self).__init__(config)
        self.__config = config
        self.__vehicles_on_screen = deque([], maxlen=int(video_fps * 2))
        self.__averages_over = int(video_fps)
        self.__processed_frames_total = 0

    def selfdiag(self):
        vbbox1 = VehicleBoundingBox(self.__config, ((20, 100), (60, 80)))
        vbbox2 = VehicleBoundingBox(self.__config, ((20, 100), (60, 80)))
        vbbox3 = VehicleBoundingBox(self.__config, ((70, 100), (110, 80)))
        vbbox4 = VehicleBoundingBox(self.__config, ((120, 100), (160, 80)))

        assert (vbbox1.vec_distance(vbbox2) == 0)
        assert (vbbox1.vec_distance(vbbox3) == 50.0)
        assert (vbbox4.vec_distance(vbbox3) == 50.0)

        vos1 = VehiclesOnScreen(self.__config)
        vos1.append(vbbox1)
        vos2 = VehiclesOnScreen(self.__config)
        vos2.append(vbbox2)
        vos3 = VehiclesOnScreen(self.__config)
        vos3.append(vbbox3)
        vos4 = VehiclesOnScreen(self.__config)
        vos4.append(vbbox4)

        self.append(vos1)
        self.append(vos2)
        self.append(vos3)
        self.append(vos4)
        self.append(vos1)
        self.append(vos1)

        test_image = 'data/test_images/test1.jpg'
        test_image = 'data/test_videos/thumbs/projectvid_29.jpg'
        image_orig = self.load_image(test_image)
        image = np.copy(image_orig)
        for bbox in self.get_all_drawables():
            print(bbox)
            image = self.__draw_boxes(image, [bbox])
            self.display_image_grid('vehicle-detection/pipeline', 'vcollection.png', [image_orig, image], ['image', 'image_final'], cmap='gray')

    def __draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        for bbox in bboxes:
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)

        return img

    def append(self, v_on_s_new):
        assert (type(v_on_s_new) is VehiclesOnScreen)
        self.__processed_frames_total += 1

        if len(self.__vehicles_on_screen) == 0:
            v_on_s_new.display_all()
            self.__vehicles_on_screen.append(v_on_s_new)
        else:
            frame_counter = 0
            for v_on_s_prev in self.__vehicles_on_screen:
                for vbbox_prev in v_on_s_prev.vbboxes():
                    for vbbox_new in v_on_s_new.vbboxes():
                        dist = vbbox_prev.vec_distance(vbbox_new)
                        if dist <= self.MIN_CENTROID_DISTANCE:
                            #if vbbox_new.width() > self.MIN_CENTROID_DISTANCE:
                            self.__logger.info("bbox good: {} dist: {} bbox id: {}".format(vbbox_new.get_bbox(), dist, vbbox_new.get_id()))
                            vbbox_new.set_id(vbbox_prev.get_id())
                            vbbox_new.set_can_display(True)
                            #else:
                            #    self.__logger.info("bbox too small: {} dist: {} bbox id: {}".format(vbbox_new.get_bbox(), dist, vbbox_new.get_id()))
                        else:
                            self.__logger.info("bbox too far: {} dist: {} id: {}".format(vbbox_new.get_bbox(), dist, vbbox_new.get_id()))


                frame_counter += 1
            self.__vehicles_on_screen.append(v_on_s_new)

    def get_all_drawables(self, min_centroids_thresh=MIN_CENTROIDS_THRESHOLD):
        centroids = {}
        for v_on_s_prev in self.__vehicles_on_screen:
            for vbbox_prev in v_on_s_prev.vbboxes():
                if vbbox_prev.can_display():
                    id = vbbox_prev.get_id()
                    if not id in centroids:
                        centroids[id] = [1, [vbbox_prev]]
                    else:
                        centroids[id][0] += 1
                        centroids[id][1].append(vbbox_prev)

        bboxes = []
        for id in centroids.keys():
            if centroids[id][0] >= min_centroids_thresh:
                self.__logger.info("min_centroids_thresh met: {}".format(min_centroids_thresh))
                p1_sum = np.array((0.0,0.0))
                p2_sum = np.array((0.0,0.0))
                weight = 1.0
                for vbbox in centroids[id][1]:
                    assert(type(vbbox) is VehicleBoundingBox)
                    p1, p2 = vbbox.get_bbox()
                    self.__logger.info("\t p1: {} p2: {}".format(p1, p2))
                    p1_sum += weight * np.array(p1)
                    p2_sum += weight * np.array(p2)

                p1_sum = p1_sum / centroids[id][0]
                p2_sum = p2_sum / centroids[id][0]

                p1 = p1_sum.astype(int)
                p2 = p2_sum.astype(int)
                self.__logger.info("\t averaged p1: {} p2: {}".format(p1, p2))
                bboxes.append (( (p1[0], p1[1]), (p2[0], p2[1]) ))

        return bboxes

class VehiclesOnScreen(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        super(VehiclesOnScreen, self).__init__(config)
        self.__config = config
        self.__vbboxes = []

    def append(self, vbbox):
        assert(type(vbbox) is VehicleBoundingBox)
        self.__vbboxes.append(vbbox)

    def vbboxes(self):
        return self.__vbboxes

    def display_all(self):
        for vbbox in self.__vbboxes:
            vbbox.set_can_display(True)

    def copy(self):
        config = self.__config
        self.__config = None
        cpy = copy.deepcopy(self)
        cpy.__config = config
        return cpy


class VehicleBoundingBox(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config, bbox):
        assert (type(bbox) is tuple)
        super(VehicleBoundingBox, self).__init__(config)
        self.__config = config
        self.__bbox = bbox
        self.__display = False
        self.__id = str(uuid.uuid4())

        lleft, uright = bbox
        midx = int((lleft[0] + uright[0]) / 2)
        midy = int((lleft[1] + uright[1]) / 2)

        self.__bbox_center = (midx, midy)

    def get_id(self):
        return self.__id

    def set_id(self, v):
        self.__id = v

    def get_bbox(self):
        return self.__bbox

    def can_display(self):
        return self.__display

    def set_can_display(self, v):
        self.__display = v

    def width(self):
        p1, p2 = self.__bbox
        return abs(p1[0] - p2[0])

    def height(self):
        p1, p2 = self.__bbox
        return abs(p1[1] - p2[1])

    def vec_distance(self, v_bbox):
        x1, y1 = self.__bbox_center
        x2, y2 = v_bbox.__bbox_center
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def copy(self):
        config = self.__config
        self.__config = None
        cpy = copy.deepcopy(self)
        cpy.__config = config
        return cpy
