"""
This script processes images of MoCA (Montreal Cognitive Assessment) tests to extract the clock drawing part.
It reads PNG images from a specified directory, detects the clock drawing using contour detection, extracts
the clock region, rotates it to the correct orientation, resizes it to a standard size, and saves the extracted
region as a new image in a specified output directory.

Usage:
    python script.py --path_to_PNG <path_to_PNG_directory> --path_to_saved_ROI <path_to_output_directory>
"""

import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Set up argument parser for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--path_to_PNG', type=str, required=True, help='Path to the directory containing PNG files.')
parser.add_argument('--path_to_saved_ROI', type=str, required=True, help='Path to the directory to save the extracted ROI images.')
args = parser.parse_args()

# Initialize list to hold detected contours
detected_contours = []

# Desired dimensions for the extracted ROI image
desired_width = 420
desired_height = 450

# Create named windows for displaying images
cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)

# Check if the output directory exists, create if it does not
folder_exist = os.path.exists(args.path_to_saved_ROI)
if not folder_exist:
    os.makedirs(args.path_to_saved_ROI)
    print("A new directory to save the ROI images has been created!")

# Loop through all the PNG files in the specified directory
for file in os.listdir(args.path_to_PNG):
    if file.endswith(".png"):
        # Split the file name and extension
        file_name, extension = os.path.splitext(file)
        
        # Read the image and convert to grayscale
        img = cv2.imread(args.path_to_PNG + '/' + file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram of the grayscale image (for analysis, not used further)
        histogram = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

        # Apply binary thresholding
        ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        
        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Loop through each contour to find the desired one
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Check if contour meets the area and perimeter criteria
            if area > 220000 and perimeter < 2500:
                detected_contours.append(contour)
                
                # Get minimum area rectangle for the contour
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get bounding box and angle of rotation
                x, y, w, h = cv2.boundingRect(box)
                center, angle = rect[0], rect[2]
                
                # Adjust angle if necessary
                if angle > 10:
                    angle = angle + 270
        
        # Create rotation matrix and rotate the image
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
        
        # Extract the Region of Interest (ROI) from the rotated image
        before_rot = img[y + 75:y + h - 65, x + 15:x + w - 15]
        roi = rotated_image[y + 75:y + h - 65, x + 15:x + w - 15]
        
        # Resize the ROI to the desired dimensions
        resized_roi = cv2.resize(roi, (desired_width, desired_height))
        
        # Draw contours on the original image for visualization
        cv2.drawContours(img, detected_contours, -1, (0, 255, 0), 3)
        
        # Display images in their respective windows
        cv2.imshow('Img', img)
        cv2.imshow('ROI', resized_roi)
        cv2.imshow('Thresh', thresh)
        cv2.imshow('Gray', img_gray)
        
        # Save the rotated image with the detected ROI
        cv2.imwrite(args.path_to_saved_ROI + '/' + file_name + '.png', rotated_image)
        
        # Wait for a key press to move to the next image
        cv2.waitKey(0)
        
        # Clear detected contours list for next iteration
        detected_contours = []

        # Optional: Display histogram (commented out)
        # plt.figure()
        # plt.title('Histogram of Grayscale Image')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.plot(histogram, color='black')
        # plt.xlim([0, 256])
        # plt.ylim([0, 100000])  # Set the y-axis limit here
        # plt.show()
