import logging
import os
import cv2
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm


class ImageProcessing(object):
    def __init__(self, config):
        self.__config = config
        self.__test_images_directory = config.get('test', 'test_images_directory')
        self.__output_images_dir = config.get('detection', 'output_directory')
        self.__data_dir = config.get('camera_calibration', 'data_directory')

    def test_files(self):
        image_files = [
         #'harder_challenge_vid_13.jpg',
         'straight_lines1.jpg',
         'straight_lines2.jpg',
         'test1.jpg',
         'test2.jpg',
         'test3.jpg',
         'test4.jpg',
         'test5.jpg',
         'test6.jpg',
         'challenge_vid_5.jpg',
         'harder_challenge_vid_2.jpg',
         'harder_challenge_vid_4.jpg',
         'harder_challenge_vid_6.jpg',
         'harder_challenge_vid_8.jpg',
         'harder_challenge_vid_9.jpg',
         'harder_challenge_vid_12.jpg',
         'harder_challenge_vid_13.jpg',
         'harder_challenge_vid_14.jpg']

        for image_file in image_files:
            yield [image_file, "{0}/{1}".format(self.__test_images_directory, image_file)]

    def selfdiag(self):
        for filename, file in self.test_files():
            img = self.load_image(file)

            gray = self.to_grayscale(img)
            assert(gray is not None)

            gray_r = self.to_grayscale(img, to_binary=True, chan='All', threshold=(180, 255))
            assert(gray_r is not None)

            hls = self.to_hls(img)
            assert (hls is not None)

            hls_s = self.to_hls(img, to_binary=True, threshold=(100, 255))
            assert (hls_s is not None)

            hsv = self.to_hsv(img)
            assert (hsv is not None)

            hsv_s = self.to_hsv(img, to_binary=True, threshold=(100, 255))
            assert (hsv_s is not None)

            for filename, file in self.test_files():
                image = self.load_image(file)
                assert(image is not None)
                masked = self.region_of_interest(image)
                #self.display_image(masked)

            #self.display_image_grid('harder_challenge_vid_14.jpg', [gray, hls, hsv], ['gray', 'hls', 'hsv'], cmap='gray')
            #self.display_image_grid('harder_challenge_vid_14.jpg', [gray_r, hls_s, hsv_s], ['gray r', 'hls s', 'hsv s'], cmap='gray')
            break

    def display_image(self, image, cmap=None):
        plt.imshow(image, cmap=cmap)
        plt.show()

    def display_image_grid(self, filename, images, titles, cmap=None, save=False):
        # Plot the result
        fig, axn = plt.subplots(1, len(images), figsize=(24, 9))
        fig.tight_layout()

        for i in range(len(images)):
            axn[i].imshow(images[i], cmap=cmap)
            axn[i].set_title(titles[i], fontsize=40)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        if save == True:
            fig.savefig("{0}/thresholded/{1}".format(self.__output_images_dir, filename))
            pass
        else:
            plt.show()

    def region_of_interest(self, image):
        image_height = image.shape[0]
        image_width = image.shape[1]

        viewport_x_min = 175
        viewport_x_max = 1075
        viewport_y_max = image_height - 55
        viewport_center_x = image_width / 2
        viewport_lane_horizon_y = 425

        vertices = np.array([[(viewport_x_min, viewport_y_max),
                              (viewport_center_x - 100, viewport_lane_horizon_y),
                              (viewport_center_x + 150, viewport_lane_horizon_y),
                              (viewport_x_max, viewport_y_max)]], dtype=np.int32)

        mask = np.zeros_like(image)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def apply_on_test_images(self):
        raise NotImplementedError("apply_on_test_images() def must be overridden!")

    def process(self, filename, image):
        raise NotImplementedError("process() def must be overridden!")

    def load_image(self, path):
        return cv2.imread(path)

    def to_grayscale(self, image, to_binary=False, chan='R', threshold=(0, 255)):
        """If you are reading in an image using mpimg.imread() this will read in an RGB image and you should convert to grayscale
           using cv2.COLOR_RGB2GRAY, but if you are using cv2.imread() or the glob API, as happens in this video example,
           this will read in a BGR image and you should convert to grayscale using cv2.COLOR_BGR2GRAY"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if to_binary == True:
            if chan == 'R':
                R = gray[:, :, 0]
                chan_selected = R
            elif chan == 'G':
                G = gray[:, :, 1]
                chan_selected = G
            elif chan == 'B':
                B = gray[:, :, 2]
                chan_selected = B
            else:
                chan_selected = gray

            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return gray

    def to_hsv(self, image, to_binary=False, chan='S', threshold=(0, 255)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if to_binary == True:
            H = hsv[:, :, 0]
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]

            chan_selected = H
            if chan == 'H':
                chan_selected = H
            elif chan == 'S':
                chan_selected = S
            elif chan == 'V':
                chan_selected = V

            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return hsv

    def to_hls(self, image, to_binary=False, chan='S', threshold=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        if to_binary == True:
            H = hls[:, :, 0]
            L = hls[:, :, 1]
            S = hls[:, :, 2]

            chan_selected = S
            if chan == 'H':
                chan_selected = H
            elif chan == 'L':
                chan_selected = L
            elif chan == 'S':
                chan_selected = S

            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return hls

    def gaussian_blur(self, image, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def to_color_binary(self, binary_image):
        color_binary = np.dstack((np.zeros_like(binary_image), binary_image, binary_image))
        return color_binary

    def __thresholded_binary_image(self, image, threshold):
        binary = np.zeros_like(image)
        binary[(image > threshold[0]) & (image <= threshold[1])] = 1
        return binary

class ImageCannyEdgeDetection(ImageProcessing):
    def __init__(self, config):
        super(ImageCannyEdgeDetection, self).__init__(config)
        self.__canny_low_threshold = config.getfloat('detection', 'canny_low_threshold')
        self.__canny_high_threshold = config.getfloat('detection', 'canny_high_threshold')
        self.__hough_threshold = config.getint('detection', 'hough_threshold')
        self.__hough_min_line_len = config.getint('detection', 'hough_min_line_len')
        self.__hough_max_line_gap = config.getint('detection', 'hough_max_line_gap')

    def selfdiag(self):
        self.apply_on_test_images()

    def apply_on_test_images(self):
        for filename, file in self.test_files():
            image = self.load_image(file)
            image = self.process(filename, image)

    def process(self, filename, image):
        image = self.to_hls(image, to_binary=False, chan='S', threshold=(180, 255))
        img_gray = self.to_grayscale(image)

        img_gaussian = self.gaussian_blur(img_gray, 3)
        img_canny_edge = self.canny(img_gaussian)

        img_masked = self.region_of_interest(img_canny_edge)

        hough_lines_image = self.hough_lines(img_masked, rho=2, theta=np.pi / 180, threshold=self.__hough_threshold, min_line_len=self.__hough_min_line_len, max_line_gap=self.__hough_max_line_gap, y_min=0)
        self.display_image_grid("canny_{0}".format(filename), [image, img_masked, hough_lines_image], ['hls', 'canny', 'hough'], cmap='gray', save=True)
        return hough_lines_image

    def canny(self, img):
        """Applies the Canny transform"""
        return cv2.Canny(img, self.__canny_low_threshold, self.__canny_high_threshold)


    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap, y_min, line_fit=False):
        """ `img` should be the output of a Canny transform. Returns an image with hough lines drawn."""

        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
        self.draw_lines(line_img, lines, y_min, line_fit)
        return line_img

    def draw_lines(self, img, lines, y_min, line_fit = False, color=[255, 0, 0], thickness=10):
        if lines is None:
            return img

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


class ImageThresholding(ImageProcessing):
    def __init__(self, config):
        super(ImageThresholding, self).__init__(config)

    def selfdiag(self):
        self.apply_on_test_images()

    def apply_on_test_images(self):
        for filename, file in self.test_files():
            image = self.load_image(file)
            image = self.process(filename, image)

    def process(self, filename, image):
        image_conv_hls = self.to_hls(image, to_binary=True, chan='S', threshold=(180, 255))
        image_conv_gray = self.to_grayscale(image)  # self.to_hls(image, to_binary=True, chan='S')
        ksize = 3

        image_conv = self.region_of_interest(image_conv_gray)

        gradx = self.abs_sobel_thresh(image_conv, orient='x', sobel_kernel=ksize, thresh=(20, 255))
        grady = self.abs_sobel_thresh(image_conv, orient='y', sobel_kernel=ksize, thresh=(20, 255))
        mag_binary = self.mag_thresh(image_conv, sobel_kernel=ksize, mag_thresh=(20, 255))
        dir_binary = self.dir_threshold(image_conv, sobel_kernel=ksize, thresh=(0.7, 1.3))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        self.display_image_grid(filename, [image_conv_gray, image_conv, gradx, grady, mag_binary, dir_binary, combined], ['gray', 'hls (s chan)', 'gradx', 'grady', 'mag_binary', 'dir_binary', 'combined'], cmap='gray', save=True)
        return combined

    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output


    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

