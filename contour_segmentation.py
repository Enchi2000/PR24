import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--path_to_roi',type=str,required=True)
parser.add_argument('--path_to_save_contour',type=str,required=True)
args=parser.parse_args()

cv2.namedWindow('Img',cv2.WINDOW_NORMAL)

h_accumulated=np.zeros((256,1),dtype=np.float32)
detected_contours=[]
outer_contours=[]
hull_detected=[]
distances=[]

def cal_histogram(img):
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    return hist

folder_exist=os.path.exists(args.path_to_save_contour)
if not folder_exist:
    os.makedirs(args.path_to_save_contour)
    print("A new directory to save the CONTOUR images has been created!")

for file in os.listdir(args.path_to_roi):
    if file.endswith(".png"):
        file_name,extension=os.path.splitext(file)
        img=cv2.imread(args.path_to_roi+'/'+file)
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask=np.zeros_like(img_gray)
        blurred=cv2.GaussianBlur(img_gray,(3,3),0)
        laplacian=cv2.Laplacian(blurred,cv2.CV_16S)
        sharp_image = cv2.convertScaleAbs(laplacian-blurred)
        ret,img_threshold=cv2.threshold(sharp_image,229,251,cv2.THRESH_BINARY_INV)
        contours,hierarchy=cv2.findContours(img_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if hierarchy[0][i][3]==-1:
                outer_contours.append(contours[i])
        for contour in outer_contours:
            area=cv2.contourArea(contour)
            perimeter=cv2.arcLength(contour,True)
            if area>1000:
                hull=cv2.convexHull(contour,returnPoints=True)
                M=cv2.moments(hull)
                detected_contours.append(contour)
                hull_detected.append(hull)
        centerx = int(M['m10'] / M['m00'])
        centery = int(M['m01'] / M['m00'])
        center = (centerx, centery)
        for point in hull:
            hull_point=tuple(point[0])
            distance=np.sqrt((centerx - hull_point[0])**2 + (centery - hull_point[1])**2)
            distances.append(distance)
            #print(f"Distance to hull point {hull_point}: {distance}")
            cv2.line(img, center, hull_point, (0, 255, 0), 2)
        cv2.circle(img, center, 2, (0, 255, 0), -1)
        kernel = np.ones((5, 5), np.uint8)
        cv2.drawContours(img,detected_contours,-1,(0,0,255),1)
        #cv2.drawContours(img,hull_detected,-1,(0,255,0),1)
        cv2.drawContours(mask,hull_detected,-1,(255,255,255),thickness=cv2.FILLED)
        gradient = cv2.erode(mask,kernel,iterations=1)
        result=mask-gradient
        mean_distance=np.mean(distance)
        print(mean_distance)
        threshold=mean_distance*2
        for i,point in enumerate(hull):
            hull_point=tuple(point[0])
            distance=distances[i]
            if distance>threshold:
                new_x = int(hull_point[0] - (hull_point[0] - centerx) * threshold / distance)
                new_y = int(hull_point[1] - (hull_point[1] - centery) * threshold / distance)
                hull[i][0][0] = new_x
                hull[i][0][1] = new_y 
        cv2.polylines(img, [hull], isClosed=True, color=(255, 255, 0), thickness=2)
        #test=cv2.bitwise_and(img,img,mask=result)
        #hist=cal_histogram(img_gray)
        #h_accumulated=h_accumulated+hist
        cv2.imshow('Img',img)
        #cv2.imshow('test',test)
        cv2.waitKey(0)
        detected_contours=[]
        outer_contours=[]
        hull_detected=[]

cv2.destroyAllWindows()

# plt.figure()
# plt.plot(hist,color='red')
# plt.xlim([0,256])
# plt.show()