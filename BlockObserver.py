"""

Block detection and analysis.

"""


import cv2
import numpy as np
import re
from colorfilters import HSVFilter


class BlockObserver:
    def __init__(self):
        self.img = None                 # Original image
        self.img_result = None          # Cropped and labelled image
        self.sequence = None            # Ordered list containing the colors found
        self.label = None               # String label that represents univocally the sequence
        # Known colors and their ranges (lower and upper)
        self.colors = {
            'blue': [
                np.array([92, 48, 20], np.uint8),
                np.array([120, 250, 255], np.uint8)
            ],
            'orange': [
                np.array([12, 130, 235], np.uint8),
                np.array([25, 255, 255], np.uint8)
            ],
            'red': [
                np.array([0, 100, 111], np.uint8),
                np.array([3, 255, 255], np.uint8)
            ],
            'green': [
                np.array([36, 70, 25], np.uint8),
                np.array([70, 255, 255], np.uint8)
            ]
        }

    # Executes all the chain of computations
    # Returns: (list: ordered colors, string: label, bool: validity)
    def process(self, img):
        self.img = img
        self.crop()
        sequence, label = self.detect_sequence()
        validity = self.validate_sequence()
        return sequence, label, validity

    # Crops the input image to observe only the building area
    def crop(self, debug=False):
        roi = self.img[280:332, 355:461]        # y1:y2, x1:x2
        roi = cv2.resize(roi, (roi.shape[1]*5, roi.shape[0]*5)) # zoom (for visual inspection)
        self.img = roi
        if debug:
            cv2.imshow("Cropped ROI", self.img)
            cv2.waitKey(0)

    # Tries to find the colored cubes and analyses their positions
    def detect_sequence(self, dilate=False, erode=True, blur=False, kernel_size=3, debug=False):
        assert self.img is not None, "Invalid operation. Invoke process() first"
        img = self.img  # Safe copy
        # converts frame to HSV
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask_dict = {}
        kernel = np.ones((kernel_size, kernel_size), "uint8")
        # Creates the binary mask and optionally morphs it
        for color, range_bounds in self.colors.items():
            mask = cv2.inRange(hsv, range_bounds[0], range_bounds[1])
            if dilate:
                mask = cv2.dilate(mask, kernel)
            if erode:
                mask = cv2.erode(mask, kernel)
            if blur:
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask_dict[color] = mask
            # Debug: displays the binary mask for the specific colors
            if debug:
                cv2.imshow(color + " mask", cv2.bitwise_and(img, img, mask=mask))
                cv2.waitKey(0)
        # Finds the centroids
        centroids = {}
        for color, mask in mask_dict.items():
            nonzero = cv2.findNonZero(mask)     # Finds all the non-black pixels and calculates a bounding rectangle
            if nonzero is None:
                continue    # If the color is not found, skip the centroid search
            nonzero = np.asmatrix(nonzero)
            # Min and max coordinates
            min_x = np.min(nonzero[:, 0])
            min_y = np.min(nonzero[:, 1])
            max_x = np.max(nonzero[:, 0])
            max_y = np.max(nonzero[:, 1])
            # Calculate centroid coordinates
            centroid_coordinates = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
            centroids[color] = centroid_coordinates
            # Draws the bounding box
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 255, 255), 1)
        # Marks the centroids on the image, for display purposes
        for color, coords in centroids.items():
            cv2.circle(img, (coords[0], coords[1]), 1, (0, 0, 0), -1)
            cv2.putText(img, color[0], (coords[0]-10, coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        self.img_result = img
        # Sort the centroids to determine the color sequence
        sequence = [x[0].upper() for x in sorted(centroids.items(), key=lambda item: item[1])]
        # Reverts the sequence
        sequence.reverse()
        self.sequence = sequence
        # Converts the list of strings in a string code (e.g. 'BORG')
        label = ""
        for element in self.sequence:
            label += element[0]
        self.label = label
        return sequence, label

    # Verifies if the cubes are aligned in a valid sequence.
    def validate_sequence(self):
        assert self.label is not None, "Invalid operation. Invoke process() first"
        # Tries to validate the string code against a regexp
        p = re.compile('([B|O][G|R][B|O][G|R])|([G|R][B|O][G|R][B|O])')
        m = p.match(self.label)
        if m:
            return True
        else:
            return False

    # Validates a partial sequence composed of only 2 blocks.
    def validate_partial_sequence(self):
        assert self.label is not None, "Invalid operation. Invoke process() first"
        # Extracts the partial label
        partial_label = self.label[:2]
        # Tries to validate the string code against a regexp
        p = re.compile('([B|O][G|R])|([G|R][B|O])')
        m = p.match(partial_label)
        if m:
            return True
        else:
            return False

    def display(self):
        cv2.imshow("Processed image", self.img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Debug: manual visual HSV filtering
    def hsv_filtering(self, img):
        self.img = img
        self.crop()
        window = HSVFilter(self.img)
        print("Press Q or ESC to close the window and print details.")
        window.show()
        print("Image filtered in HSV between {" + str(window.lowerb) + "} and {" + str(window.upperb) + "}.")
