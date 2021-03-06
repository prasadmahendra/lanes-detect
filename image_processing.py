import logging
import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import math

class ImageProcessing(object):
    __logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.__config = config
        self.__test_images_directory = config.get('test', 'test_images_directory')
        self.__output_images_dir = config.get('detection', 'output_directory')
        self.__data_dir = config.get('camera_calibration', 'data_directory')
        self.__save_output_images = config.getboolean('global', 'save_output_images')

    def test_files(self):
        image_files = [
         'projectvid_1.jpg',
         'projectvid_2.jpg',
         'projectvid_3.jpg',
         'straight_lines1.jpg',
         'straight_lines2.jpg',
         'test1.jpg',
         'test2.jpg',
         'test3.jpg',
         'test4.jpg',
         'test5.jpg',
         'test6.jpg',
         'challenge_vid_5.jpg']

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
                masked = self.region_of_interest(image, binary_image=False)
                assert (masked is not None)
                masked = self.region_of_interest(image, binary_image=True)
                assert (masked is not None)
                #self.display_image(masked)

            break

    def save_output_images(self):
        return self.__save_output_images

    def display_image(self, image, cmap=None):
        plt.imshow(image, cmap=cmap)
        plt.show()

    def display_image_grid(self, subfolder, filename, images, titles, cmap=None, save=False):
        plt.close()
        output_images_enabled = self.__config.getboolean('global', 'output_images_enabled')
        if output_images_enabled == False:
            return

        rows = math.ceil(len(images) / 4)

        fig, axn = plt.subplots(rows, min(len(images), 4), figsize=(24, 9))
        fig.tight_layout()

        for i in range(len(images)):
            if rows == 1:
                axn[i].imshow(images[i], cmap=cmap)
                axn[i].set_title(titles[i], fontsize=40)
            else:
                y = math.floor(i / 4)
                x = i % 4
                axn[y][x].imshow(images[i], cmap=cmap)
                axn[y][x].set_title(titles[i], fontsize=40)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.suptitle(filename)

        if save == True or self.save_output_images() == True:
            self.__logger.info("saving {0}/{1}/{2}".format(self.__output_images_dir, subfolder, filename))
            fig.savefig("{0}/{1}/{2}".format(self.__output_images_dir, subfolder, filename))
        else:
            plt.show()

    def engine_compart_pixes(self):
        return 50

    def region_of_interest(self, image, binary_image):
        image_height = image.shape[0]
        image_width = image.shape[1]

        viewport_x_min = 175
        viewport_x_max = 1150
        viewport_y_max = image_height - self.engine_compart_pixes()
        viewport_center_x = image_width / 2
        viewport_lane_horizon_y = 425

        vertices = np.array([[(viewport_x_min, viewport_y_max),
                              (viewport_center_x - 100, viewport_lane_horizon_y),
                              (viewport_center_x, viewport_lane_horizon_y),
                              (viewport_center_x, viewport_lane_horizon_y + 50),
                              (viewport_x_min + 250, viewport_y_max),
                              (viewport_x_max - 250, viewport_y_max),
                              (viewport_center_x, viewport_lane_horizon_y + 50),
                              (viewport_center_x, viewport_lane_horizon_y),
                              (viewport_center_x + 150, viewport_lane_horizon_y),
                              (viewport_x_max, viewport_y_max)]], dtype=np.int32)

        mask = np.zeros_like(image)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            if binary_image:
                ignore_mask_color = (1,) * channel_count
            else:
                ignore_mask_color = (255,) * channel_count
        else:
            if binary_image:
                ignore_mask_color = 1
            else:
                ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def apply_on_test_images(self):
        raise NotImplementedError("apply_on_test_images() def must be overridden!")

    def process(self, image, filename=None, display=False):
        raise NotImplementedError("process() def must be overridden!")

    def load_image(self, path):
        img = cv2.imread(path)
        assert(img is not None)
        return img

    def to_grayscale(self, image, to_binary=False, chan='ALL', threshold=(0, 255)):
        """If you are reading in an image using mpimg.imread() this will read in an RGB image and you should convert to grayscale
           using cv2.COLOR_RGB2GRAY, but if you are using cv2.imread() or the glob API, as happens in this video example,
           this will read in a BGR image and you should convert to grayscale using cv2.COLOR_BGR2GRAY"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        chan = chan.upper()
        if chan != 'ALL':
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
        else:
            chan_selected = gray

        if to_binary == True:
            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return chan_selected

    def to_hsv(self, image, to_binary=False, chan='ALL', threshold=(0, 255)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        chan = chan.upper()
        if chan != 'ALL':
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
        else:
            chan_selected = hsv

        if to_binary == True:
            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return chan_selected

    def to_luv(self, image, to_binary=False, chan='ALL', threshold=(0, 255)):
        luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        chan = chan.upper()
        if chan != 'ALL':
            L = luv[:, :, 0]
            U = luv[:, :, 1]
            V = luv[:, :, 2]

            chan_selected = L
            if chan == 'L':
                chan_selected = L
            elif chan == 'U':
                chan_selected = U
            elif chan == 'V':
                chan_selected = V
        else:
            chan_selected = luv

        if to_binary == True:
            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return chan_selected

    def to_yuv(self, image, to_binary=False, chan='ALL', threshold=(0, 255)):
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        chan = chan.upper()
        if chan != 'ALL':
            Y = yuv[:, :, 0]
            U = yuv[:, :, 1]
            V = yuv[:, :, 2]

            chan_selected = Y
            if chan == 'Y':
                chan_selected = Y
            elif chan == 'U':
                chan_selected = U
            elif chan == 'V':
                chan_selected = V
        else:
            chan_selected = yuv

        if to_binary == True:
            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return chan_selected

    def to_ycrcb(self, image, to_binary=False, chan='ALL', threshold=(0, 255)):
        YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        chan = chan.upper()
        if chan != 'ALL':
            Y = YCrCb[:, :, 0]
            Cr = YCrCb[:, :, 1]
            Cb = YCrCb[:, :, 2]

            chan_selected = Y
            if chan == 'Y':
                chan_selected = Y
            elif chan == 'CR':
                chan_selected = Cr
            elif chan == 'CB':
                chan_selected = Cb
        else:
            chan_selected = YCrCb

        if to_binary == True:
            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return chan_selected

    def to_hls(self, image, to_binary=False, chan='ALL', threshold=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        chan = chan.upper()
        if chan != 'ALL':
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
        else:
            chan_selected = hls

        if to_binary == True:
            return self.__thresholded_binary_image(chan_selected, threshold)
        else:
            return chan_selected

    def gaussian_blur(self, image, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def to_color_binary(self, binary_image):
        color_binary = np.dstack((np.zeros_like(binary_image), binary_image, binary_image))
        return color_binary

    def fig2data(self, fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf

    def __thresholded_binary_image(self, image, threshold):
        binary = np.zeros_like(image)
        binary[(image > threshold[0]) & (image <= threshold[1])] = 1
        return binary

    def unzip(self, zippedfile, outfolder):
        with zipfile.ZipFile(zippedfile) as zf:
            for member in zf.infolist():
                if not member.filename.startswith("__MACOSX"):
                    self.__logger.info("Extracting {0}/{1}".format(outfolder, member.filename))
                    zf.extract(member, outfolder)

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
            image = self.process(image, filename)

    def process(self, image, filename=None, display=False):
        #image = self.to_hls(image, to_binary=False, chan='S', threshold=(180, 255))
        img_gray = self.to_grayscale(image)

        img_gaussian = self.gaussian_blur(img_gray, 3)
        img_canny_edge = self.canny(img_gaussian)

        img_masked = self.region_of_interest(img_canny_edge, binary_image=False)

        hough_lines_image = self.hough_lines(img_masked, rho=2, theta=np.pi / 180, threshold=self.__hough_threshold, min_line_len=self.__hough_min_line_len, max_line_gap=self.__hough_max_line_gap, y_min=0)

        self.display_image_grid("thresholded", "canny_{0}".format(filename), [image, img_masked, hough_lines_image], ['hls', 'canny', 'hough'], cmap='gray', save=self.save_output_images())

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
            image = self.process(image, filename)

    def process(self, image, filename=None, display=True):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # extract yellow ....
        yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))

        # extract white ....
        sensitivity_1 = 68
        white_1 = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

        sensitivity_2 = 60
        hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        white_2 = cv2.inRange(hsl, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
        white_3 = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
        white = white_1 | white_2 | white_3

        hls_2 = self.to_hls(image, to_binary=True, chan='S', threshold=(90, 255))
        image_conv = hls_2 | yellow | white

        ksize = 3

        gradx = self.abs_sobel_thresh(image_conv, orient='x', sobel_kernel=ksize, thresh=(20, 255))
        grady = self.abs_sobel_thresh(image_conv, orient='y', sobel_kernel=ksize, thresh=(20, 255))
        mag_binary = self.mag_thresh(image_conv, sobel_kernel=ksize, mag_thresh=(20, 255))
        dir_binary = self.dir_threshold(image_conv, sobel_kernel=ksize, thresh=(0.7, 1.3))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        if display:
            self.display_image_grid('thresholded', filename,
                                    [image, hls_2, yellow, white, gradx, grady, mag_binary, dir_binary, combined],
                                    ['image', 'hls_2', 'yellow', 'white', 'gradx', 'grady', 'mag_binary', 'dir_binary', 'combined'], cmap='gray', save=self.save_output_images())

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

