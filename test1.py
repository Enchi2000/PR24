import cv2
import numpy as np
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--path_to_hands',type=str,required=True)
args=parser.parse_args()

# Trackbar callback function
def update_parameters(dummy=None):
    # Get current trackbar positions
    threshold = cv2.getTrackbarPos('Threshold', 'Parameters')
    min_line_length = cv2.getTrackbarPos('Min Line Length', 'Parameters')
    max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Parameters')

    # Run Hough line detection with updated parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Draw detected lines on the original image
    if lines is not None:
        image_with_lines = np.copy(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the result with detected lines
        cv2.imshow('Parameters', image_with_lines)
    else:
        # If no lines detected, display the original image
        cv2.imshow('Parameters', image)

for file in os.listdir(args.path_to_hands):
    #Check if files end with .PNG
    if file.endswith(".png"):
        file_name,extension=os.path.splitext(file)
        image=cv2.imread(args.path_to_hands+'/'+file)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Create a window for trackbars
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)

        # Create trackbars for adjusting parameters
        cv2.createTrackbar('Threshold', 'Parameters', 35, 500, update_parameters)
        cv2.createTrackbar('Min Line Length', 'Parameters', 9, 200, update_parameters)
        cv2.createTrackbar('Max Line Gap', 'Parameters', 10, 100, update_parameters)

        # Initialize parameters
        update_parameters()

        # Wait for any key to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()

