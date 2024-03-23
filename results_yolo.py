from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import random

# Load a model
model = YOLO('/home/enchi/Documentos/Biggest_Dataset.pt')  # pretrained YOLOv8n model

def update_params(*args):
    # Get current trackbar positions
    source_copy = source.copy()
    dp = cv2.getTrackbarPos('dp', 'Parameters')
    minDist = cv2.getTrackbarPos('minDist', 'Parameters')
    param1 = cv2.getTrackbarPos('param1', 'Parameters')
    param2 = cv2.getTrackbarPos('param2', 'Parameters')
    minRadius = cv2.getTrackbarPos('minRadius', 'Parameters')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Parameters')

    # Perform Hough Circle Transform
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # Draw circles on the source image if any are detected
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(source_copy, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circle, (x, y), r, (255), -1)
    else:
        print('No circles detected.')

    # Display the source image with circles
    cv2.imshow('Parameters', source_copy)

def near_contour(contours,point,point1):
    min_distance = np.inf
    number=None
    nearest_contour=None
    point_contour=None

    for i,cnt in enumerate(contours):
        area=cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            contour_centroid_x = int(M["m10"] / M["m00"])
            contour_centroid_y = int(M["m01"] / M["m00"])
            dist = np.sqrt((point[0] - contour_centroid_x)**2 + (point[1] - contour_centroid_y)**2)
            if area>20 and area<500:  # Ensure the point is inside the contour and update the nearest contour
                min_distance = dist
                nearest_contour = cnt
                number=i
                point_contour=contour_centroid_x,contour_centroid_y

    return number,nearest_contour,point_contour

def distance(centroid1, centroid2):
    x1, y1 = centroid1
    x2, y2 = centroid2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_centroid(bbox):
    x1, y1, w, h = bbox
    return (int(x1 + w / 2), int(y1 + h / 2))

def draw_bounding_boxes(image, bboxes, color=(255, 255, 0), thickness=1):
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)



for file in os.listdir('/home/enchi/Documentos/PEF/test_images'):
    #Check if files end with .PNG
    if file.endswith(".png"):
        file_name,extension=os.path.splitext(file)
        source=cv2.imread('/home/enchi/Documentos/PEF/test_images'+'/'+file)
        clock_contour=cv2.imread('/home/enchi/Imágenes/Adjust/'+file)
        clock_contour=cv2.cvtColor(clock_contour,cv2.COLOR_BGR2GRAY)
        draw=source.copy()
        gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
        draw1=source.copy()
        mask = np.zeros_like(gray)
        numbers = np.zeros_like(gray)
        circle=np.zeros_like(gray)
        

        initial_area = cv2.countNonZero(clock_contour)

        # Define erosion rate (e.g., 10% reduction in area)
        erosion_rate = 0.0

        # Initialize variables
        current_area = initial_area
        eroded_mask = clock_contour.copy()
        count=0

        while current_area > initial_area * (1 - erosion_rate):
            # Define structuring element for erosion
            kernel = np.ones((5,5),np.uint8)
            
            # Perform erosion
            eroded_mask = cv2.erode(eroded_mask, kernel, iterations=1)
            
            # Calculate the area of the eroded mask
            current_area = cv2.countNonZero(eroded_mask)
        clock_contour=eroded_mask

        M=cv2.moments(clock_contour)
        #Calculate the center of contour
        centerx = int(M['m10'] / M['m00'])
        centery = int(M['m01'] / M['m00'])
        center = (centerx, centery)

        contours, _ = cv2.findContours(clock_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(draw1,contours,-1,(0,255,0),1) 
        if contours:
            # cv2.drawContours(draw1,contours,-1,(0,255,0),1)
            contour=contours[0]
            points_on_contours=contour[:,0]
            radius_list=[]
            selected_points=[]
            angles=[0, 30, 60, 90, 120, 150, 180,210, 240, 270,300,330]
            for point in points_on_contours:
                point_x, point_y = point
                angle_deg = np.degrees(np.arctan2(centery-point_y, point_x - centerx))
                # Ensure angle is positive
                if angle_deg < 0:
                    angle_deg += 360
                # Check if the angle is close to a multiple of 30 degrees
                if int(angle_deg) in angles:
                    angles.remove(int(angle_deg))
                    # If yes, add this point to the selected points list
                    selected_points.append((point,int(angle_deg)))  
            nearest_contours_points=[] 
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours=list(contours)
            for point, angle in selected_points:
                if angle in [90]:
                    limit_x_max=point[0]*1.2
                    limit_x_min=point[0]*0.8
                    limit_y_max=point[1]*1.5
                    limit_y_min=point[1]*0.4
                    bboxes=[]
                    condition=True
                    iterations=contours
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h))
                            
                            ratio=w/h
                            area_r=w*h
                            cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1) 
                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max:   
                                if ratio>1.2:
                                    if 300<=area_r<1400:
                                        cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 0, 255), 1)
                                        # bboxes.append((x,y,w,h))
                                        cv2.circle(draw1,(middle_x,middle_y),2,(0,0,255),-1)
                                        condition=False
                                else:
                                    # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1)
                                    if 48<=area_r<1300:
                                        bboxes.append((x,y,w,h))
                                        cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                    else:
                                        h=int(h*0.5)
                                        if h>w:
                                            # bboxes.append((x,y,w,h))
                                            cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                          
                        else:
                            condition=False
                        if len(iterations)==0:
                            print('lol')

                    # Keep track of the closest pair of bounding boxes
                    closest_pair = None
                    min_separation = float('inf')
                    closest_bboxes=[]
                    min_dist1 = float('inf')
                    min_dist2 = float('inf')

                    # Iterate over each pair of bounding boxes
                    for i in range(len(bboxes)):
                        centroid1 = calculate_centroid(bboxes[i])
                        x,y,w,h=bboxes[i]
                        for j in range(i+1, len(bboxes)):
                            x1,y1,w1,h1=bboxes[j]
                            final_w=w+w1
                            f_ratio=final_w/h
                            dist_y=abs(y-y1)
                            centroid2 = calculate_centroid(bboxes[j])
                            separation = distance(centroid1, centroid2)
                            if separation < min_separation and f_ratio>0.6 and dist_y<7:
                                min_separation = separation
                                closest_pair = (i, j)
                                closest_bboxes=[bboxes[i],bboxes[j]]
                    draw_bounding_boxes(draw1, closest_bboxes)



                cv2.circle(draw1,tuple(point),2,(0,0,255),-1)

            # cv2.drawContours(draw1,contours,-1,(0,255,0),1)



        #YOLO---------------------------------------------------------------------------------------------
        # results = model(source, conf=0.7, iou=0.7,max_det=200,imgsz=448)
        # xyxys = []
        # confidences = []
        # class_ids = []
        # centroids=[]
        # roi_list=[]

        # for result in results:
        #     boxes = result.boxes.cpu().numpy()
        #     xyxys.extend(boxes.xyxy)  # Use extend to append multiple values
        #     confidences.extend(boxes.conf)
        #     class_ids.extend(boxes.cls)
        #     for i, xyxy in enumerate(xyxys):
        #         x1, y1, x2, y2 = map(int, xyxy)
        #         class_name = class_ids[i]
        #         # Assuming confidences contain confidence scores
        #         confidence = confidences[i]
        #         label = f'{class_name}: {confidence:.2f}'
        #         centroid_x = (x1 + x2) // 2
        #         centroid_y = (y1 + y2) // 2
        #         centroids.append((centroid_x, centroid_y))
        #         if class_name in [1,2,3,4,5,6,7,8,9,10,11,12]:     
        #             cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255,0), 1)
        #             cv2.putText(draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #         roi=gray[y1:y2,x1:x2]
        #         roi_height, roi_width = roi.shape[:2]
        #         contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #         for cnt in contours:
        #             contour_area = cv2.contourArea(cnt)
        #             if contour_area>50:
        #                 x, y, w, h = cv2.boundingRect(cnt)
        #                 # cv2.rectangle(draw, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (0, 0, 255), 1)
        #         numbers[y1:y2,x1:x2]=roi
        #         cv2.circle(mask, (centroid_x, centroid_y), 2, 255, -1)
        #------------------------------------------------------------------------------------------

        
        # cv2.namedWindow('Parameters',cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Parameters", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # subtracted_image = cv2.bitwise_and(gray, cv2.bitwise_not(numbers))
        # subtracted_image = cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2RGB)

        # Define default values for Hough Circle Transform parameters
        default_dp = 4
        default_minDist = 320
        default_param1 = 50
        default_param2 = 30
        default_minRadius = 60
        default_maxRadius = 143
        # Create trackbars for each parameter
        # cv2.createTrackbar('dp', 'Parameters', default_dp, 10, update_params)
        # cv2.createTrackbar('minDist', 'Parameters', default_minDist, 400, update_params)
        # cv2.createTrackbar('param1', 'Parameters', default_param1, 100, update_params)
        # cv2.createTrackbar('param2', 'Parameters', default_param2, 100, update_params)
        # cv2.createTrackbar('minRadius', 'Parameters', default_minRadius, 100, update_params)
        # cv2.createTrackbar('maxRadius', 'Parameters', default_maxRadius, 400, update_params)

        # Call update_params initially to perform Hough Circle Transform with default parameters
        # update_params()
        # outliers=cv2.bitwise_and(circle,mask)
        

        # cv2.imshow('test', numbers)
        cv2.namedWindow('source',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('source', 800, 600)  # Set the width and height of the window
        cv2.imshow('source',draw1)
        # cv2.imshow('current',eroded_mask)
        # cv2.imshow('initial',clock_contour)
        # cv2.imshow('test',test)
        # cv2.imshow('circle',draw1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()