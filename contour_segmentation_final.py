# Usage:
# This script extracts the clock contour of the test Montreal Cognitive Assessment (MoCA)
# using morphological operations and then detects the lines representing the clock hands.

# Description:
# The Montreal Cognitive Assessment (MoCA) is a widely used screening tool for detecting
# cognitive impairment. One of its tasks involves drawing a clock showing a specific time.
# This script is designed to process images of the clock drawn by individuals taking the MoCA
# test. It applies morphological operations to extract the clock contour and then detects
# the lines representing the clock hands.

# Import necessary libraries
import os
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path_to_roi', type=str, required=True)
parser.add_argument('--path_to_save_contour', type=str, required=True)
args = parser.parse_args()

# Initialize variables
detected_inner_contours = []
detected_outer_contours = []
outer_contours = []
hull_detected = []
inner_contours = []
final_shape = []
direction_change_points = []
points = []
min_distance = float('inf')

def group_angles_with_tolerance(angles_lengths):
    """
    Group angles within a certain tolerance to account for similar directions.

    Args:
    - angles_lengths: List of tuples containing angle, length pairs.

    Returns:
    - grouped_angles: List of tuples containing grouped angle, length pairs.
    """
    grouped_angles = []
    current_group = []
    x_group = []
    y_group = []

    # Iterate through sorted list of angles_lengths
    for i in range(len(angles_lengths)):
        angle, length, x, y = angles_lengths[i]

        # Check if there's a current group and if the angle is within tolerance of the last angle in the group
        if current_group and abs(angle - current_group[-1][0]) <= 0.15 * current_group[-1][0]:
            current_group.append((angle, length))
            x_group.append(x)
            y_group.append(y)
        else:
            # Start a new group
            if current_group:
                avg_angle = sum(angle for angle, _ in current_group) / len(current_group)
                avg_length = sum(length for _, length in current_group) / len(current_group)
                x1_avg = sum(x_group) / len(x_group)
                y1_avg = sum(y_group) / len(y_group)
                grouped_angles.append((avg_angle, avg_length, x1_avg, y1_avg))
            current_group = [(angle, length)]
            x_group = [x]
            y_group = [y]

    # Add the last group
    if current_group:
        avg_angle = sum(angle for angle, _ in current_group) / len(current_group)
        avg_length = sum(length for _, length in current_group) / len(current_group)
        x1_avg = sum(x_group) / len(x_group)
        y1_avg = sum(y_group) / len(y_group)
        grouped_angles.append((avg_angle, avg_length, x1_avg, y1_avg))

    return grouped_angles

# Create a directory to save contour images if it does not exist
folder_exist = os.path.exists(args.path_to_save_contour)
if not folder_exist:
    os.makedirs(args.path_to_save_contour)
    print("A new directory to save the contour images has been created!")

# Read files in the ROI directory
for file in os.listdir(args.path_to_roi):
    # Check if files end with .PNG
    if file.endswith(".png"):
        # Split file name into file, extension
        file_name, extension = os.path.splitext(file)
        # Read image
        img = cv2.imread(args.path_to_roi + '/' + file)
        # Get the height and width of image
        height, width, _ = img.shape
        # Transform image into grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create mask for future bitwise operations
        mask = np.zeros_like(img_gray)
        mask1 = np.zeros_like(img_gray)
        mask2 = np.zeros_like(img_gray)
        mask3 = np.zeros_like(img_gray)
        mask4 = np.zeros_like(img_gray)
        hands = np.zeros_like(img_gray)
        
        # Apply Gaussian Filter to grayscale image
        blurred = cv2.blur(img_gray, (3, 3))
        # Apply Laplacian filter to blurred image
        laplacian = cv2.Laplacian(blurred, cv2.CV_16S)
        # Make the image sharper
        sharp_image = cv2.convertScaleAbs(laplacian - blurred)
        laplacian = cv2.convertScaleAbs(laplacian)
        histogram = cv2.calcHist([sharp_image], [0], None, [256], [0, 256])
        
        # Apply binarization to image
        ret, img_threshold = cv2.threshold(sharp_image, 229, 255, cv2.THRESH_BINARY_INV)

        # Find contours in image
        contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Initialize lists to store outer and inner contours
        outer_contours = []
        inner_contours = []

        # Iterate through contours and classify them as outer or inner based on hierarchy
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                outer_contours.append(contours[i])
            if hierarchy[0][i][3] != -1:
                inner_contours.append(contours[i])

        # Initialize lists to store detected inner and outer contours and their convex hulls
        detected_inner_contours = []
        detected_outer_contours = []
        hull_detected = []

        # Iterate through inner contours and add those with area greater than 1000 to detected_inner_contours
        for contour in inner_contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                detected_inner_contours.append(contour)

        # Iterate through outer contours and add those with area greater than 1000 to detected_outer_contours
        for contour in outer_contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                hull = cv2.convexHull(contour, returnPoints=True)
                detected_outer_contours.append(contour)
                hull_detected.append(hull)

        # Define kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)

        # If both inner and outer contours are detected
        if detected_inner_contours and detected_outer_contours:
            # Draw inner contours in mask
            cv2.drawContours(mask, detected_inner_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            # Draw outer contours in mask
            cv2.drawContours(mask1, detected_outer_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            # Apply bitwise not to outer contours to detect outside the region
            mask1 = cv2.bitwise_not(mask1)
            # Make white area bigger of inner and outside the outer contour
            inner_contour_dilated = cv2.dilate(mask, kernel, iterations=3)
            outer_contour_dilated = cv2.dilate(mask1, kernel, iterations=3)
            # Where areas meet it should be the contour of the image
            result = cv2.bitwise_and(outer_contour_dilated, inner_contour_dilated)

        # If only outer contours are detected
        else:
            # Draw the hull contour of outer contours
            result = cv2.drawContours(mask1, hull_detected, -1, (255, 255, 255), thickness=cv2.FILLED)
            # Make hull bigger for better detection of contour
            result = cv2.dilate(result, kernel, iterations=1)
            # Make the hull smaller
            adjust = cv2.erode(result, kernel, iterations=4)
            # Subtract the adjust to result to obtain only the contour
            result = result - adjust

        # Close any open contour
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        # Find contours in the result
        contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Initialize list to store final shape contours
        final_shape = []

        # Iterate through contours and find those with no parent (outermost contours)
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                hull = cv2.convexHull(contour, returnPoints=True)
                final_shape.append(hull)

        # Draw final contours on mask2
        cv2.drawContours(mask2, final_shape, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Erode mask2 to remove noise and adjust contour size
        adjust = cv2.erode(mask2, kernel, iterations=4)
        mask2 = mask2 - adjust
        other = cv2.bitwise_not(mask2)

        # Apply bitwise operations to obtain final contour and rest of the image
        final_contour = cv2.bitwise_and(img, img, mask=mask2)
        rest1 = cv2.bitwise_and(img, img, mask=other)

        # Make 3 channel masks for final contour and rest of the image
        three_channel_mask = cv2.merge([mask2] * 3)
        three_channel_mask1 = cv2.merge([other] * 3)

        # Combine final contour and rest of the image with respective masks
        rest = rest1 + three_channel_mask
        final_contour = final_contour + three_channel_mask1

        
        # Get the moments from mask2
        M = cv2.moments(mask2)

        # Calculate the center of contour
        centerx = int(M['m10'] / M['m00'])
        centery = int(M['m01'] / M['m00'])
        center = (centerx, centery)

        # Draw a circle at the center of the clock contour (optional)
        # cv2.circle(img, center, 3, (0, 0, 255), -1)

        # Apply bitwise operations to obtain grayscale image of the rest of the clock
        rest_gray = cv2.bitwise_and(sharp_image, sharp_image, mask=other)
        rest_gray = rest_gray + mask2
        ret, rest_gray_th = cv2.threshold(rest_gray, 229, 255, cv2.THRESH_BINARY_INV)

        # Close any gaps in clock hands using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        adjust2 = cv2.erode(adjust, kernel, iterations=30)
        closing = cv2.morphologyEx(rest_gray_th, cv2.MORPH_CLOSE, kernel)

        # Find contours in the closing image
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Initialize variables to find nearest contour to clock center
        min_distance = float('inf')
        nearest_contour = None

        # Iterate through contours to find the one nearest to the clock center
        for contour in contours:
            # Calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                contour_centroid_x = int(M["m10"] / M["m00"])
                contour_centroid_y = int(M["m01"] / M["m00"])
                # Calculate distance between contour centroid and image centroid
                distance = np.sqrt((centerx - contour_centroid_x) ** 2 + (centery - contour_centroid_y) ** 2)
                # Update nearest contour if distance is smaller and contour perimeter is large enough
                if distance < min_distance and perimeter > 100:
                    min_distance = distance
                    nearest_contour = contour

        # Create a mask containing the nearest contour
        cv2.drawContours(mask4, [nearest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Find contours again in the mask4 to detect clock hands
        contours, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Initialize variables to find the lowest point on clock hands
        lowest_point = (0, 0)
        threshold_distance = 40

        # Iterate through contours to find the lowest point on clock hands
        for contour in contours:
            test = cv2.convexHull(contour)
            for point in test:
                x, y = point[0]
                points.append((x, y))
                distance_to_centroid = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                # Update lowest point if criteria are met
                if y > lowest_point[1] and distance_to_centroid < threshold_distance:
                    lowest_point = (x, y)

        # If no lowest point is found, set it to the center of the clock
        if lowest_point == (0, 0):
            lowest_point = center

        # Apply Canny edge detection to the grayscale image of the rest of the clock
        edges = cv2.Canny(rest_gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform on the edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=37, minLineLength=22, maxLineGap=10)

        # Initialize list to store angles and lengths of detected lines
        angles_lengths = []

        # Draw the detected lines on the original image and store their angles and lengths
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate Euclidean distance between starting point and point of interest
                distance = np.sqrt((x1 - lowest_point[0]) ** 2 + (y1 - lowest_point[1]) ** 2)
                distance1 = np.sqrt((x2 - lowest_point[0]) ** 2 + (y2 - lowest_point[1]) ** 2)

                # Add lines with appropriate angles and lengths to the list
                if distance < 30:  # Adjust the threshold as needed
                    angle = np.arctan2(y1 - y2, x2 - x1) * 180.0 / np.pi
                    if angle < 0:
                        angle += 360.0
                    if angle > 10 and angle < 70:
                        angles_lengths.append((angle, distance1, x1, y1))
                if distance1 < 30:
                    angle = np.arctan2(y2 - y1, x1 - x2) * 180.0 / np.pi
                    # If angle is negative, make it positive
                    if angle < 0:
                        angle += 360.0
                    if angle > 90 and angle < 160:
                        angles_lengths.append((angle, distance, x2, y2))

            # Sort the list of angles and lengths
            angles_lengths.sort(key=lambda x: x[0])

            # Group angles with similar orientation
            grouped_angles = group_angles_with_tolerance(angles_lengths)

            # Draw the lines on the original image based on grouped angles and lengths
            for angle, length, x_A, y_A in grouped_angles:
                # Calculate the endpoint of the line based on the average angle and length
                x2 = int(x_A + length * np.cos(np.radians(angle)))
                y2 = int(y_A - length * np.sin(np.radians(angle)))

                # Draw the line on the image
                cv2.line(img, (int(x_A), int(y_A)), (x2, y2), (0, 0, 255), 2)


        # Show the original image with drawn lines
        cv2.imshow('Img', img)
        # Uncomment below to show additional images
        # cv2.imshow('contour', final_contour)
        # cv2.imshow('rest', sharp_image)
        # cv2.imshow('numbers', numbers)
        # cv2.imshow('hand_clock', adjust)

        # Uncomment below to plot histogram
        # plt.figure()
        # plt.title('Histogram of Grayscale Image')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.plot(histogram, color='black')
        # plt.xlim([0, 256])
        # plt.ylim([0, 1000])  # Set the y-axis limit here
        # plt.show()

        # Uncomment below to save the image into the directory
        # cv2.imwrite(args.path_to_save_contour+'/'+file_name+'.png', img)

        # Wait until a key is pressed
        cv2.waitKey(0)

        # Reset all variables for the next iteration
        detected_outer_contours = []
        detected_inner_contours = []
        outer_contours = []
        hull_detected = []
        inner_contours = []
        final_shape = []
        direction_change_points = []
        points = []
        min_distance = float('inf')

# Close all windows
cv2.destroyAllWindows()


