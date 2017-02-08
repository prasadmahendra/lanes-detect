import cv2
import numpy as np
import multiprocessing
import logging
import concurrent.futures
from scipy.ndimage.measurements import label
from vehicle_detection import VehicleDetection
from vehicle import VehiclesCollection, VehiclesOnScreen, VehicleBoundingBox

class VehicleSearch(VehicleDetection):
    __logger = logging.getLogger(__name__)

    def __init__(self, config, video_fps=30):
        super(VehicleSearch, self).__init__(config)

        self.__config = config
        self.__y_start_pos_pix = config.getint('vehicle_search', 'y_start_pos_pix')
        self.__y_stop_pos_pix = config.getint('vehicle_search', 'y_stop_pos_pix')
        self.__x_start_pos_pix = config.getint('vehicle_search', 'x_start_pos_pix')
        self.__x_stop_pos_pix = config.getint('vehicle_search', 'x_stop_pos_pix')
        self.__heatmap_threshold_value = config.getint('vehicle_search', 'heatmap_threshold')
        self.__classifier, self.__scaler = self.load_classifier()
        self.__vehicles_collection = VehiclesCollection(config, video_fps)

    def selfdiag(self):
        self.__selfdiag2()

    def __selfdiag2(self):
        image_files = [
            'projectvid_28.jpg',
            'projectvid_29.jpg',
            'projectvid_30.jpg',
            'projectvid_31.jpg',
            'projectvid_32.jpg',
            'projectvid_33.jpg',
            'projectvid_34.jpg',
            'projectvid_35.jpg',
            'projectvid_36.jpg',
            'projectvid_37.jpg',
            'projectvid_38.jpg',
            'projectvid_39.jpg',
            'projectvid_40.jpg',
            'projectvid_41.jpg',
            'projectvid_42.jpg',
            'projectvid_43.jpg',
            'projectvid_44.jpg',
            'projectvid_45.jpg',
            'projectvid_46.jpg',
            'projectvid_47.jpg',
            'projectvid_48.jpg',
            'projectvid_49.jpg',
            'projectvid_50.jpg',
            'projectvid_51.jpg',
            'projectvid_52.jpg',
            'projectvid_53.jpg'
        ]

        for counter in range(50):
            image = self.load_image("data/test_videos/thumbs/projectvid_{}.jpg".format(counter + 1))
            image_final = self.process(image)
            self.display_image_grid('vehicle-detection/pipeline', 'pipeline-{}.png'.format(counter), [image, image_final], ['image', 'image_final'], cmap='gray')

    def __selfdiag1(self):
        for counter in range(50):
            test_image = 'data/test_images/test1.jpg'
            #test_image = "data/test_videos/thumbs/projectvid_{}.jpg".format(counter + 1)
            image = self.load_image(test_image)

            image_roi = self.__draw_boxes(np.copy(image), [[(self.__x_start_pos_pix,self.__y_stop_pos_pix), (self.__x_stop_pos_pix, self.__y_start_pos_pix)]])

            windows = self.__slide_all_windows(image)
            car_windows, heatmap = self.__detect_vehicle_windows(image, windows)
            heatmap_thresholded = self.__heatmap_threshold(np.copy(heatmap), threshold=self.__heatmap_threshold_value)
            cars_bboxes_img = self.__draw_boxes(np.copy(image), car_windows)

            labels = label(heatmap_thresholded)
            self.__logger.info('{} cars found'.format(labels[1]))
            image_labelled = self.__draw_labeled_bboxes(np.copy(image), labels)
            bboxes = self.__labels_to_bbox(labels)
            cars_bboxes_img_final = self.__draw_boxes(np.copy(image), bboxes)

            self.display_image_grid('vehicle-detection/sliding-windows', 'sliding-windows-roi-{}.png'.format(counter), [image, image_roi, cars_bboxes_img], ['image', 'search region', 'cars_bboxes_img'], cmap='gray')
            self.display_image_grid('vehicle-detection/sliding-windows', 'sliding-windows-roi-heatmaps-{}.png'.format(counter), [image, heatmap, heatmap_thresholded, image_labelled, cars_bboxes_img_final], ['image', 'heatmap', 'heatmap_thresholded', 'image_labelled', 'final'], cmap='gray')
            break

    def process(self, image):
        windows = self.__slide_all_windows(image)
        car_windows, heatmap = self.__detect_vehicle_windows(image, windows)
        heatmap_thresholded = self.__heatmap_threshold(np.copy(heatmap), threshold=self.__heatmap_threshold_value)

        labels = label(heatmap_thresholded)
        bboxes = self.__labels_to_bbox(labels)

        vos = VehiclesOnScreen(self.__config)
        for bbox in bboxes:
            vbbox = VehicleBoundingBox(self.__config, bbox)
            vos.append(vbbox)

        self.__vehicles_collection.append(vos)

        cars_bboxes_img_final = image
        for bbox in self.__vehicles_collection.get_all_drawables():
            cars_bboxes_img_final = self.__draw_boxes(image, [bbox])
            self.display_image_grid('vehicle-detection/pipeline', 'vcollection.png', [image, image], ['image', 'image_final'], cmap='gray')

        return cars_bboxes_img_final

    def __detect_vehicle_windows(self, image, windows):
        image_height = image.shape[0]
        image_width = image.shape[1]

        heatmap = np.zeros((image_height, image_width), dtype=np.uint8)

        # TODO: extract hog once vs. per each sliding window

        # 1) Create an empty list to receive positive detection windows
        on_windows = []

        # 2) Iterate over all windows in the list
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            all_futs = {executor.submit(self.__predict_on_window, window, image, on_windows, heatmap): window for window in windows}
            for future in concurrent.futures.as_completed(all_futs):
                window = all_futs[future]
                try:
                    (_, prediction) = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (window, exc))
                else:
                    if prediction[0] == 1:
                        on_windows.append(window)
                        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        # 8) Return windows for positive detections
        return on_windows, heatmap

    def __predict_on_window(self, window, image, found_windows, heatmap):
        # 3) Extract the test window from original image
        test_img = self.resize_for_classification(image[window[0][1]:window[1][1], window[0][0]:window[1][0]])
        # 4) Extract features for that window using single_img_features()
        features = self.extract_hog_features(test_img)
        # 5) Scale extracted features to be fed to classifier
        test_features = self.__scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = self.__classifier.predict(test_features)

        #predict_proba = self.__classifier.predict_proba([features])[0]
        #self.__logger.info("predict_proba: {}".format(predict_proba))
        #if prediction[0] == 1:
        #    self.__logger.info("--> {}: {}".format(prediction, predict_proba))

        if prediction[0] == 1:
            self.__logger.info("car detected. prediction: {}".format(prediction))
            return (window, prediction)
        else:
            return (window, prediction)

    def __search_bbox_sizes(self):
        return [
            #(320, 320),
            #(288, 288),
            #(256, 256),
            #(224, 224),
            (192, 192),
            #(160, 160),
            (128, 128),
            (96, 96),
            (64, 64),
            #(48, 48),
        ]

    def __slide_all_windows(self, image):
        y_start = self.__y_start_pos_pix
        y_stop = self.__y_stop_pos_pix - self.engine_compart_pixes()
        x_start = self.__x_start_pos_pix
        x_stop = self.__x_stop_pos_pix

        slide = 0
        overlap = 0.90
        window_list = []

        for bbox_size in self.__search_bbox_sizes():
            windows = self.__slide_window(image, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop], xy_window=bbox_size, xy_overlap=(overlap, overlap))

            x_start += 35
            x_stop -= 35
            y_stop -= 25
            slide += 1

            window_list += windows

        return window_list

    def __slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        image_width = img.shape[1]
        image_midpoint = image_width / 2

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step)# - 1
        ny_windows = np.int(yspan / ny_pix_per_step)# - 1

        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        counter = 0

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]

                bbox_midpoint = startx + ((endx - startx) / 2)
                deviation = abs((bbox_midpoint - image_midpoint) / image_midpoint)
                window_width_adj = int(deviation * 10)
                endx = endx + window_width_adj

                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                if endy > y_start_stop[1]:
                    continue  # bbox overflow
                if endx > x_start_stop[1]:
                    continue  # bbox overflow

                bbox = ((startx, starty), (endx, endy))

                # Append window position to list
                window_list.append(bbox)
                counter += 1


        # Return the list of windows
        return window_list

    def __draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=3):
        for bbox in bboxes:
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)

        return img

    def __heatmap_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap < threshold] = 0
        # Return thresholded map
        return heatmap

    def __labels_to_bbox(self, labels):
        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)

        # Return the image
        return bboxes

    def __draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
