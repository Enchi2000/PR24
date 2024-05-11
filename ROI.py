import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--path_to_PNG',type=str,required=True)
parser.add_argument('--path_to_saved_ROI',type=str,required=True)
args=parser.parse_args()

detected_contours=[]

desired_width=420
desired_height=450

cv2.namedWindow('Img',cv2.WINDOW_NORMAL)
cv2.namedWindow('ROI',cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray',cv2.WINDOW_NORMAL)
cv2.namedWindow('Thresh',cv2.WINDOW_NORMAL)

folder_exist=os.path.exists(args.path_to_saved_ROI)
if not folder_exist:
    os.makedirs(args.path_to_saved_ROI)
    print("A new directory to save the ROI images has been created!")
for file in os.listdir(args.path_to_PNG):
    if file.endswith(".png"):
        file_name,extension=os.path.splitext(file)  
        img=cv2.imread(args.path_to_PNG+'/'+file)
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

        ret, thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY)
        thresh=cv2.bitwise_not(thresh)
        contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            perimeter=cv2.arcLength(contour,True)
            if area>220000 and perimeter<2500:
                detected_contours.append(contour)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x,y,w,h=cv2.boundingRect(box)
                center,angle=rect[0],rect[2]
                if angle>10:
                    angle=angle+270
        rot_matrix=cv2.getRotationMatrix2D(center,angle,1)
        rotated_image=cv2.warpAffine(img,rot_matrix,(img.shape[1],img.shape[0]))
        before_rot=img[y+75:y+h-65,x+15:x+w-15]
        roi=rotated_image[y+75:y+h-65,x+15:x+w-15]
        resized_roi=cv2.resize(roi,(desired_width,desired_height))
        cv2.drawContours(img,detected_contours,-1,(0,255,0),3)
        cv2.imshow('Img',img)
        cv2.imshow('ROI',resized_roi)
        cv2.imshow('Thresh',thresh)
        cv2.imshow('Gray',img_gray)
        cv2.imwrite(args.path_to_saved_ROI+'/'+file_name+'.png',img)
        cv2.waitKey(0)
        detected_contours=[]

        # plt.figure()
        # plt.title('Histogram of Grayscale Image')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.plot(histogram, color='black')
        # plt.xlim([0, 256])
        # plt.ylim([0, 100000])  # Set the y-axis limit here
        # plt.show()