"""

Block detection and analysis.

"""


import cv2
import numpy as np
import re


class BlockObserver:
    def __init__(self):
        self.latest_image = None
        self.latest_sequence = None
        # Known colors and their ranges (lower and upper)
        self.colors = {
            'blue': [
                np.array([99, 115, 150], np.uint8),
                np.array([110, 255, 255], np.uint8)
            ],
            'orange': [
                np.array([10, 100, 200], np.uint8),
                np.array([25, 255, 255], np.uint8)
            ],
            'red': [
                np.array([136, 87, 111], np.uint8),
                np.array([180, 255, 255], np.uint8)
            ],
            'green': [
                np.array([36, 25, 25], np.uint8),
                np.array([70, 255, 255], np.uint8)
            ]
        }

    # todo crop input image?

    # Tries to find the colored cubes and analyses their positions
    def detect_sequence(self, img, dilate=False, erode=True, blur=False, kernel_size=5):
        # converts frame to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
            # Displays the binary mask for the specific color
            #cv2.imshow(color + " mask", cv2.bitwise_and(img, img, mask=mask))
            #cv2.waitKey(0)
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
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2)
        # Marks the centroids on the image, for display purposes
        for color, coords in centroids.items():
            cv2.circle(img, (coords[0], coords[1]), 10, (0, 0, 0), -1)
            cv2.putText(img, color, (coords[0]-50, coords[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), cv2.LINE_4)
        self.latest_image= img
        # Sort the centroids to determine the color sequence
        sequence = [x[0].upper() for x in sorted(centroids.items(), key=lambda item: item[1])]
        self.latest_sequence = sequence
        return sequence

    # Verifies if the cubes are aligned in a valid sequence. Optionally, analyzes a new sequence.
    def validate_sequence(self, sequence=None):
        if not sequence:
            sequence = self.latest_sequence
        # Converts the list of strings in a string code (e.g. 'BORG')
        string_sequence = ""
        for element in sequence:
            string_sequence += element[0]
        # Tries to validate the string code against a regexp
        p = re.compile('([B|O][G|R][B|O][G|R])|([G|R][B|O][G|R][B|O])')
        m = p.match(string_sequence)
        if m:
            return True
        else:
            return False

    def display(self):
        cv2.imshow("Processed image", self.latest_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
