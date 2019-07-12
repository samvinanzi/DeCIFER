"""

Color detection and blob analysis.

"""


import cv2
import numpy as np


class ColorObserver:
    def __init__(self):
        self.processed_img = None
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

    # Tries to find the colored cubes and analyses their positions
    def find_cubes(self, img, dilate=False, erode=True, blur=False, kernel_size=5):
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
            nonzero = np.asmatrix(nonzero)
            # Min
            min_x = np.min(nonzero[:, 0])
            min_y = np.min(nonzero[:, 1])
            # Max
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
        self.processed_img = img
        # Sort the centroids to determine the color sequence
        sequence = [x[0].upper() for x in sorted(centroids.items(), key=lambda item: item[1])]
        return sequence

    def display(self):
        cv2.imshow("Processed image", self.processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
