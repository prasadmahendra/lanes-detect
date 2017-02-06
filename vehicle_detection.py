import os
import urllib.request
import logging
import cv2
import glob
import pickle
import time
import json
from tqdm import tqdm
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from image_processing import ImageProcessing
from sklearn.model_selection import train_test_split

class VehicleDetection(ImageProcessing):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        super(VehicleDetection, self).__init__(config)
        self.__classifier = config.get('vehicle_detection', 'classifier')
        self.__vehicles_training_data_src = config.get('vehicle_detection', 'vehicles_training_data')
        self.__non_vehicles_training_data_src = config.get('vehicle_detection', 'non_vehicles_training_data')
        self.__training_data_folder = config.get('vehicle_detection', 'training_data_folder')
        self.__vehicles_training_data = "{0}/vehicles.zip".format(self.__training_data_folder)
        self.__non_vehicles_training_data = "{0}/non_vehicles.zip".format(self.__training_data_folder)
        self.__model_image_input_height = config.getint('vehicle_detection', 'training_image_height')
        self.__model_image_input_width = config.getint('vehicle_detection', 'training_image_width')

        self.__hog_colorspace = config.get('hog_feature_extraction', 'colorspace')
        self.__hog_orient = config.getint('hog_feature_extraction', 'orient')
        self.__hog_pix_per_cell = config.getint('hog_feature_extraction', 'pix_per_cell')
        self.__hog_cell_per_block = config.getint('hog_feature_extraction', 'cell_per_block')
        self.__hog_channel = config.get('hog_feature_extraction', 'channel')

        self.__colorspace_spatial_size = tuple(json.loads(config.get('color_feature_extraction', 'spatial_size')))
        self.__colorspace_hist_bins = config.getint('color_feature_extraction', 'hist_bins')
        self.__colorspace_hist_range = tuple(json.loads(config.get('color_feature_extraction', 'hist_range')))

        self.__feature_extraction_type = config.get('vehicle_detection', 'feature_extraction_type')
        self.__include_color_features = False
        if self.__feature_extraction_type == "both":
            self.__include_color_features = True
        self.__classifier_obj = None

    def selfdiag(self):
        self.__logger.info("Training ...")
        self.__logger.info("Feature extraction: {}".format(self.__feature_extraction_type))
        (vehicle_images, non_vehicle_images) = self.__download_train_dataset()

        assert (vehicle_images is not None)
        assert (non_vehicle_images is not None)

        car = self.load_image(vehicle_images[0])
        nocar = self.load_image(non_vehicle_images[0])

        assert (car is not None)
        assert (nocar is not None)

        features, gray_hog = self.__get_hog_features(img=self.to_grayscale(car, to_binary=False),
                                                     orient=self.__hog_orient,
                                                     pix_per_cell=self.__hog_pix_per_cell,
                                                     cell_per_block=self.__hog_pix_per_cell,
                                                     vis=True)

        features, h_hog = self.__get_hog_features(img=self.to_hls(car, to_binary=False, chan='H'),
                                                     orient=self.__hog_orient,
                                                     pix_per_cell=self.__hog_pix_per_cell,
                                                     cell_per_block=self.__hog_pix_per_cell,
                                                     vis=True)

        features, l_hog = self.__get_hog_features(img=self.to_hls(car, to_binary=False, chan='L'),
                                                     orient=self.__hog_orient,
                                                     pix_per_cell=self.__hog_pix_per_cell,
                                                     cell_per_block=self.__hog_pix_per_cell,
                                                     vis=True)

        features, s_hog = self.__get_hog_features(img=self.to_hls(car, to_binary=False, chan='S'),
                                                     orient=self.__hog_orient,
                                                     pix_per_cell=self.__hog_pix_per_cell,
                                                     cell_per_block=self.__hog_pix_per_cell,
                                                     vis=True)

        self.display_image_grid('vehicle-detection/hog-features', 'car-hog-features.png', [car, gray_hog, h_hog, l_hog, s_hog], ['car image', 'gray_hog', 'h_hog', 'l_hog', 's_hog'], cmap='gray')

        features, gray_hog = self.__get_hog_features(img=self.to_grayscale(nocar, to_binary=False),
                                                     orient=self.__hog_orient,
                                                     pix_per_cell=self.__hog_pix_per_cell,
                                                     cell_per_block=self.__hog_pix_per_cell,
                                                     vis=True)

        features, h_hog = self.__get_hog_features(img=self.to_hls(nocar, to_binary=False, chan='H'),
                                                  orient=self.__hog_orient,
                                                  pix_per_cell=self.__hog_pix_per_cell,
                                                  cell_per_block=self.__hog_pix_per_cell,
                                                  vis=True)

        features, l_hog = self.__get_hog_features(img=self.to_hls(nocar, to_binary=False, chan='L'),
                                                  orient=self.__hog_orient,
                                                  pix_per_cell=self.__hog_pix_per_cell,
                                                  cell_per_block=self.__hog_pix_per_cell,
                                                  vis=True)

        features, s_hog = self.__get_hog_features(img=self.to_hls(nocar, to_binary=False, chan='S'),
                                                  orient=self.__hog_orient,
                                                  pix_per_cell=self.__hog_pix_per_cell,
                                                  cell_per_block=self.__hog_pix_per_cell,
                                                  vis=True)

        self.display_image_grid('vehicle-detection/hog-features', 'nocar-hog-features.png', [nocar, gray_hog, h_hog, l_hog, s_hog], ['nocar image', 'gray_hog', 'h_hog', 'l_hog', 's_hog'], cmap='gray')

    def predict(self, image):
        if self.__classifier_obj is None:
            saved_trained_classifier = "{0}/saved_trained_classifier.pickle".format(self.__training_data_folder)
            self.__logger.info("loading saved classifier: {}".format(saved_trained_classifier))
            self.__classifier_obj = pickle.load(open(saved_trained_classifier, "rb"))

        image = cv2.resize(image, (self.__model_image_input_height, self.__model_image_input_width))
        features = self.__extract_hog_features_ex(image,
                                                  cspace=self.__hog_colorspace,
                                                  orient=self.__hog_orient,
                                                  pix_per_cell=self.__hog_pix_per_cell,
                                                  cell_per_block=self.__hog_cell_per_block,
                                                  hog_channel=self.__hog_channel,
                                                  include_color_features=self.__include_color_features)

        prediction = self.__classifier_obj.predict([features])
        return prediction[0]


    def train(self):
        self.__logger.info("Training ...")
        self.__logger.info("Feature extraction: {}".format(self.__feature_extraction_type))

        saved_features_file = "{0}/saved_features.pickle".format(self.__training_data_folder)
        saved_trained_classifier = "{0}/saved_trained_classifier.pickle".format(self.__training_data_folder)

        if not os.path.isfile(saved_features_file):
            (vehicle_images, non_vehicle_images) = self.__download_train_dataset()

            car_features = self.__extract_hog_features(vehicle_images,
                                                       cspace=self.__hog_colorspace,
                                                       orient=self.__hog_orient,
                                                       pix_per_cell=self.__hog_pix_per_cell,
                                                       cell_per_block=self.__hog_cell_per_block,
                                                       hog_channel=self.__hog_channel,
                                                       include_color_features=self.__include_color_features)

            notcar_features = self.__extract_hog_features(non_vehicle_images,
                                                          cspace=self.__hog_colorspace,
                                                          orient=self.__hog_orient,
                                                          pix_per_cell=self.__hog_pix_per_cell,
                                                          cell_per_block=self.__hog_cell_per_block,
                                                          hog_channel=self.__hog_channel,
                                                          include_color_features=self.__include_color_features)

            self.__logger.info("Saving features file: {}".format(saved_features_file))
            pickle.dump([car_features, notcar_features], open(saved_features_file, "wb"))
        else:
            self.__logger.info("loading saved features file: {}".format(saved_features_file))
            [car_features, notcar_features] = pickle.load(open(saved_features_file, "rb"))

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        # Use a linear SVC
        clf = self.__get_classifier()

        # Check the training time for the SVC
        t = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()

        self.__logger.info("Time to train: {}".format(round(t2 - t, 2)))

        # Check the score of the SVC
        self.__logger.info("Test accuracy: {}".format(round(clf.score(X_test, y_test), 4)))

        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        self.__logger.info("SVC predicts: {}".format(clf.predict(X_test[0:n_predict])))
        self.__logger.info("... for these {} labels: {}".format(n_predict, y_test[0:n_predict]))
        t2 = time.time()

        self.__logger.info("{} seconds to predict {} labels".format(round(t2 - t, 5), n_predict))

        self.__logger.info("Saving classifier: {}".format(saved_trained_classifier))
        pickle.dump(clf, open(saved_trained_classifier, "wb"))

    def __get_classifier(self):
        clf = None

        if self.__classifier == 'LinearSVC':
            self.__logger.info('Using LinearSVC classifier')
            clf = LinearSVC()
        elif self.__classifier == 'SVC':
            self.__logger.info('Using SVC classifier')
            clf = SVC(kernel='rbf', gamma='auto', C=1.0)
        elif self.__classifier == 'DecisionTreeClassifier':
            self.__logger.info('Using DecisionTreeClassifier classifier')
            clf = DecisionTreeClassifier(min_samples_split=50, criterion='entropy')
        else:
            raise Exception("Unsupported classifier {}".format(self.__classifier))

        return clf

    def __download_train_dataset(self):
        if not os.path.isfile(self.__vehicles_training_data):
            self.__logger.info("Downloading {0} -> {1} ...".format(self.__vehicles_training_data_src, self.__vehicles_training_data))
            # download the dataset from s3 ...
            urllib.request.urlretrieve(self.__vehicles_training_data_src, self.__vehicles_training_data)
            # unzip ...
            self.unzip(self.__vehicles_training_data, self.__training_data_folder)

        if not os.path.isfile(self.__non_vehicles_training_data):
            self.__logger.info("Downloading {0} -> {1}".format(self.__non_vehicles_training_data_src, self.__non_vehicles_training_data))
            # download the dataset from s3 ...
            urllib.request.urlretrieve(self.__non_vehicles_training_data_src, self.__non_vehicles_training_data)
            # unzip ...
            self.unzip(self.__non_vehicles_training_data, self.__training_data_folder)

        vehicle_images = glob.glob("{0}/vehicles/*/*.png".format(self.__training_data_folder))
        non_vehicle_images = glob.glob("{0}/non-vehicles/*/*.png".format(self.__training_data_folder))

        self.__logger.info('Training data set - vehicles: {} non-vehicles: {}'.format(len(vehicle_images), len(non_vehicle_images)))
        return (vehicle_images, non_vehicle_images)


    def __get_bin_spatial(self, img, size=(32, 32)):
        """Define a function to compute binned color features"""

        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    def __get_color_hist(self, img, nbins=32, bins_range=(0, 256)):
        """Define a function to compute color histogram features"""

        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def __get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        """Define a function to return HOG features and visualization"""

        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def __extract_color_features(self, imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
        """Define a function to extract features from a list of images. Have this function call bin_spatial() and color_hist()"""

        self.__logger.info("Extract color features. cspace: {}, spatial_sizel: {}".format(cspace, spatial_size))

        # Create a list to append feature vectors to
        features = []

        # Iterate through the list of images
        for file in tqdm(imgs):
            # Read in each one by one
            image = self.load_image(file)
            assert(image is not None)

            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            else:
                feature_image = np.copy(image)

            # Apply bin_spatial() to get spatial color features
            spatial_features = self.__get_bin_spatial(feature_image, size=spatial_size)

            # Apply color_hist() also with a color space option now
            hist_features = self.__get_color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features)))

        # Return list of feature vectors
        return features

    def __extract_hog_features(self, imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, include_color_features=False):
        """Define a function to extract features from a list of images. Have this function call get_hog_features()"""

        self.__logger.info("Extract HOG features. cspace: {}, orient: {}, pix_per_cell: {}, cell_per_block: {}, hog_channel: {} include_color_features: {}".format(cspace, orient, pix_per_cell, cell_per_block, hog_channel, include_color_features))

        if include_color_features:
            self.__logger.info("Extract color features. cspace: {}, spatial_size: {}, hist_bins: {}, hist_range: {}".format(cspace, self.__colorspace_spatial_size, self.__colorspace_hist_bins, self.__colorspace_hist_range))

        # Create a list to append feature vectors to
        features = []

        # Iterate through the list of images
        for file in tqdm(imgs):
            image = self.load_image(file)
            assert (image is not None)

            image_features = self.__extract_hog_features_ex(image, cspace, orient, pix_per_cell, cell_per_block, hog_channel, include_color_features)
            features.append(image_features)

        # Return list of feature vectors
        return features

    def __extract_hog_features_ex(self, image, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, include_color_features=False):
        feature_image = None
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = self.to_hsv(image, to_binary=False, chan='ALL')
            elif cspace == 'LUV':
                feature_image = self.to_luv(image, to_binary=False, chan='ALL')
            elif cspace == 'HLS':
                feature_image = self.to_hls(image, to_binary=False, chan='ALL')
            elif cspace == 'YUV':
                feature_image = self.to_yuv(image, to_binary=False, chan='ALL')
            elif cspace == 'YCrCb':
                feature_image = self.to_ycrcb(image, to_binary=False, chan='ALL')
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                    self.__get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False,
                                            feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.__get_hog_features(feature_image[:, :, int(hog_channel)], orient, pix_per_cell, cell_per_block,
                                                   vis=False, feature_vec=True)

        if include_color_features:
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.__get_bin_spatial(feature_image, size=self.__colorspace_spatial_size)

            # Apply color_hist() also with a color space option now
            hist_features = self.__get_color_hist(feature_image, nbins=self.__colorspace_hist_bins,
                                                  bins_range=self.__colorspace_hist_range)

            # Append the new feature vector to the features list
            return np.concatenate((spatial_features, hist_features, hog_features))
        else:
            # Append the new feature vector to the features list
            hog_features
