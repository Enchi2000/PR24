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

def near_contour(contours,point):
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
            if area>20 and area<1400:  # Ensure the point is inside the contour and update the nearest contour
                nearest_contour = cnt
                number=i
                point_contour=contour_centroid_x,contour_centroid_y

    return number,nearest_contour,point_contour

def distance(centroid1, centroid2):
    x1, y1 = centroid1
    x2, y2 = centroid2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_centroid(bbox):
    x1, y1, w, h,number= bbox
    return (int(x1 + w / 2), int(y1 + h / 2))

def draw_bounding_boxes(image, bboxes, color=(255, 255, 0), thickness=1):
    if len(bboxes)>1:
        x,y,w,h,number=bboxes[0]
        x1,y1,w1,h1,number1=bboxes[1]
        difference_x=abs(x-x1)
        difference_y=abs(y-y1)
        max_h=max(h,h1)
        if x<x1:
            real_width=difference_x+w1
            real_x=x
        else:
            real_x=x1
            real_width=difference_x+w
        if y<y1:
            real_height=difference_y+max_h
            real_y=y
        else:
            real_height=difference_y+max_h
            real_y=y1
        real_bbox=(real_x,real_y,real_width,real_height)

        cv2.rectangle(image, (real_x, real_y), (real_x + real_width, real_y + real_height), color, thickness=1)
    else:
        for bbox in bboxes:
            real_x, real_y, real_width, real_height ,number= bbox
            cv2.rectangle(image, (real_x, real_y), (real_x + real_width, real_y + real_height), color, thickness=1)
        real_bbox=(real_x,real_y,real_width,real_height)
    return real_bbox

def get_number(condition,reference):
    limit_x_max=int(point[0]+70)
    limit_x_min=int(point[0]-70)
    limit_y_max=int(point[1]+80)
    limit_y_min=int(point[1]-80)
    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
    min_dist=float('inf')
    iterations=contours.copy()
    while condition:
        number,nearest_contour,point_contour=near_contour(iterations,point)
        if number is not None or nearest_contour is not None or point_contour is not None:
            del iterations[number] 
            hull=cv2.convexHull(nearest_contour,returnPoints=True)
            perimeter = cv2.arcLength(nearest_contour, True)
            approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
            x, y, w, h = cv2.boundingRect(nearest_contour)
            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
            ratio=w/h
            area=cv2.contourArea(nearest_contour)
            area_r=w*h
            for item in numbers_detected:
                if item[1] == reference:
                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                else:
                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
            

            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                # print('Inicio')
                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                # print(ratio)
                if ratio<1.4:
                    # print(ratio)
                    dist=distance((x,y),(Pr_number_x,Pr_number_y))
                    # print(dist)
                    if y<(Pr_number_y+10) and dist<min_dist and x<Pr_number_x+30:
                        min_dist=dist
                        bboxes=[]
                        bboxes.append((x,y,w,h,number))
                    # elif bboxes==[]:
                    #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                    #         min_dist=dist
                    #         bboxes=[]
                    #         bboxes.append((x,y,w,h,number))

                    
        else:
            condition=False
    if len(iterations)==0:
        print('lol')
    if bboxes:
        number=bboxes[0][4]
        del contours[number]
        number_detected=draw_bounding_boxes(draw1,bboxes)
        numbers_detected.append((number_detected,reference+1))
    else:
        numbers_detected.append(((point[0],point[1],0,0),reference+1))
        print('Ningun 9 detectado')

    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
    return bboxes



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
        circle=np.zeros_like(gray)
        

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
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours=list(contours)
            sorted_points = sorted(selected_points, key=lambda x: x[1])
            angles_less_than_90 = [(point, angle) for point, angle in selected_points if angle <= 90]
            angles_greater_than_90 = [(point, angle) for point, angle in selected_points if angle > 90]
            # Sort the parts in descending order based on the angle
            angles_less_than_90.sort(key=lambda x: x[1], reverse=True)
            angles_greater_than_90.sort(key=lambda x: x[1],reverse=True)
            # Concatenate the sorted lists
            rearranged_list = angles_less_than_90 + angles_greater_than_90

            # print(rearranged_list)
            numbers_detected=[]
            for point, angle in rearranged_list:
                bboxes=[]
                condition=True
                if angle in [90]:
                    limit_x_max=int(point[0]*1.2)
                    limit_x_min=int(point[0]*0.8)
                    limit_y_max=int(point[1]*1.6)
                    limit_y_min=int(point[1]*0.4)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (255, 255, 0), 1) 
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area_r=w*h
                            # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1) 
                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max:   
                                if ratio>1.2:
                                    if 300<=area_r<1400:
                                        bboxes=[]
                                        # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 0, 255), 1)
                                        bboxes.append((x,y,w,h,number))
                                        # cv2.circle(draw1,(middle_x,middle_y),2,(0,0,255),-1)
                                        condition=False
                                else:
                                    # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1)
                                    if 48<=area_r<1300:
                                        bboxes.append((x,y,w,h,number))
                                        # cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                    else:
                                        h=int(h*0.5)
                                        if h>w:
                                            bboxes=[]
                                            bboxes.append((x,y,w,h,number))
                                            # cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                            condition=False   
                        else:
                            condition=False
                        if len(iterations)==0:
                            print('lol')
                    # Keep track of the closest pair of bounding boxes
                    if len(bboxes)>1:
                        closest_pair = None
                        min_separation = float('inf')
                        closest_bboxes=[]
                        numbers=None

                        # Iterate over each pair of bounding boxes
                        for i in range(len(bboxes)):
                            centroid1 = calculate_centroid(bboxes[i])
                            x,y,w,h,number_i=bboxes[i]
                            for j in range(i+1, len(bboxes)):
                                x1,y1,w1,h1,number_j=bboxes[j]
                                final_w=w+w1
                                f_ratio=final_w/h
                                dist_y=abs(y-y1)
                                centroid2 = calculate_centroid(bboxes[j])
                                separation = distance(centroid1, centroid2)
                                if separation < min_separation and f_ratio>0.6 and dist_y<7:
                                    min_separation = separation
                                    closest_pair = (i, j)
                                    closest_bboxes=[bboxes[i],bboxes[j]]
                                    numbers=(number_i,number_j)

                        if closest_bboxes:
                            number_detected=draw_bounding_boxes(draw1, closest_bboxes)
                            numbers_detected.append((number_detected,12))

                        if numbers is not None:
                            del contours[numbers[0]]
                            del contours[numbers[1]]
                        else:
                            print("Ningun 12 detectado")
            
                    elif len(bboxes)==1 and number is not None:
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,12))
                    else:
                        print("Ningun 12 detectado")
                                
                elif angle in [60]:
                    limit_x_max=int(point[0]+60)
                    limit_x_min=int(point[0]-60)
                    limit_y_max=int(point[1]+60)
                    limit_y_min=int(point[1]-60)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1) 
                    min_w=float('inf')
                    min_dist=float('inf')
                    iterations=contours.copy()
                    bboxes=[]
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area_r=w*h
                            # print(numbers_detected[0][0])
                            if numbers_detected:
                                Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=numbers_detected[0][0]
                            else:
                                Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+10,0,0,0)
                            #     cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                            # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                if ratio<1:
                                    # dist=distance(point,(middle_x,middle_y))
                                    if w<min_w and middle_x>Pr_number_x:
                                        bboxes=[]
                                        min_w=w
                                        bboxes.append((x,y,w,h,number))

                        else:
                            condition=False
                        if len(iterations)==0:
                            print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,1))
                    else:
                        print('Ningun 1 detectado')
            
            
                elif angle in [30]:
                    limit_x_max=int(point[0]+60)
                    limit_x_min=int(point[0]-80)
                    limit_y_max=int(point[1]+60)
                    limit_y_min=int(point[1]-60)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_w=float('inf')
                    min_hull=float('inf')
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            hull=cv2.convexHull(nearest_contour,returnPoints=True)
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area_r=w*h
                            for item in numbers_detected:
                                if item[1] == 1:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
  
                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                                if ratio<1.2:
                                    dist=distance(point,(middle_x,middle_y))
                                    if x>Pr_number_x and middle_y<centery and middle_y>Pr_number_y and len(hull)<min_hull and dist<min_dist:
                                        min_hull=len(hull)
                                        min_dist=dist
                                        bboxes=[]
                                        bboxes.append((x,y,w,h,number))
                                        # print(len(hull))
                                        # cv2.polylines(draw1,[hull],isClosed=True,color=150,thickness=1)
                                elif ratio>1.5 and area_r>3000:
                                    # print(area_r)
                                    x=x+w-25
                                    y=y+10
                                    w=25
                                    h=25
                                    # cv2.rectangle(draw1, (x,y), (x + w, y + h), color=(255,0,150), thickness=1)
                                    bboxes=[]
                                    bboxes.append((x,y,w,h,number))
                                    condition=False
                            # else:
                            #     cv2.rectangle(draw1, (x,y), (x + w, y + h), color=(255,0,150), thickness=1)


                        else:
                            condition=False
                    if len(iterations)==0:
                        print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,2))
                    else:
                        print('Ningun 2 detectado')
                elif angle in [0]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_w=float('inf')
                    min_hull=float('inf')
                    min_dist=float('inf')
                    max_h=0
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            hull=cv2.convexHull(nearest_contour,returnPoints=True)
                            perimeter = cv2.arcLength(nearest_contour, True)
                            approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area=cv2.contourArea(nearest_contour)
                            area_r=w*h
                            for item in numbers_detected:
                                if item[1] == 2:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            

                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                                if ratio<1.2:
                                    dist=distance(point,(middle_x,middle_y))
                                    if y>Pr_number_y and middle_y<=centery and middle_x>Pr_number_x and h>max_h:
                                        max_h=h
                                        bboxes=[]
                                        bboxes.append((x,y,w,h,number))
                                    elif bboxes==[]:
                                        if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                                            min_dist=dist
                                            bboxes=[]
                                            bboxes.append((x,y,w,h,number))

                                    
                        else:
                            condition=False
                    if len(iterations)==0:
                        print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,3))
                    else:
                        print('Ningun 3 detectado')

                    # cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                
                elif angle in [330]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            hull=cv2.convexHull(nearest_contour,returnPoints=True)
                            perimeter = cv2.arcLength(nearest_contour, True)
                            approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area=cv2.contourArea(nearest_contour)
                            area_r=w*h
                            for item in numbers_detected:
                                if item[1] == 3:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            

                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                                if ratio<1.2:
                                    dist=distance((x,y),(Pr_number_x,Pr_number_y))
                                    # print(dist)
                                    if y>Pr_number_y and middle_y>centery and dist<min_dist:
                                        min_dist=dist
                                        bboxes=[]
                                        bboxes.append((x,y,w,h,number))
                                    # elif bboxes==[]:
                                    #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                                    #         min_dist=dist
                                    #         bboxes=[]
                                    #         bboxes.append((x,y,w,h,number))

                                    
                        else:
                            condition=False
                    if len(iterations)==0:
                        print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,4))
                    else:
                        print('Ningun 4 detectado')

                    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                elif angle in [300]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            hull=cv2.convexHull(nearest_contour,returnPoints=True)
                            perimeter = cv2.arcLength(nearest_contour, True)
                            approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area=cv2.contourArea(nearest_contour)
                            area_r=w*h
                            for item in numbers_detected:
                                if item[1] == 4:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            

                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                                if ratio<1.2:
                                    dist=distance((x,y),(Pr_number_x,Pr_number_y))
                                    # print(dist)
                                    if y>Pr_number_y and middle_y>centery and dist<min_dist and x<Pr_number_x:
                                        min_dist=dist
                                        bboxes=[]
                                        bboxes.append((x,y,w,h,number))
                                    # elif bboxes==[]:
                                    #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                                    #         min_dist=dist
                                    #         bboxes=[]
                                    #         bboxes.append((x,y,w,h,number))

                                    
                        else:
                            condition=False
                    if len(iterations)==0:
                        print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,5))
                    else:
                        print('Ningun 5 detectado')

                    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                elif angle in [270]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            hull=cv2.convexHull(nearest_contour,returnPoints=True)
                            perimeter = cv2.arcLength(nearest_contour, True)
                            approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area=cv2.contourArea(nearest_contour)
                            area_r=w*h
                            for item in numbers_detected:
                                if item[1] == 5:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            

                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                                # print(ratio)
                                if ratio<1.4:
                                    dist=distance((x,y),(Pr_number_x,Pr_number_y))
                                    # print(dist)
                                    if y>(Pr_number_y-10) and middle_y>centery and dist<min_dist and x<Pr_number_x:
                                        min_dist=dist
                                        bboxes=[]
                                        bboxes.append((x,y,w,h,number))
                                    # elif bboxes==[]:
                                    #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                                    #         min_dist=dist
                                    #         bboxes=[]
                                    #         bboxes.append((x,y,w,h,number))

                                    
                        else:
                            condition=False
                    if len(iterations)==0:
                        print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,6))
                    else:
                        print('Ningun 6 detectado')
                        numbers_detected.append(((point[0],point[1],0,0),6))
                        

                    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)

                elif angle in [240]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            hull=cv2.convexHull(nearest_contour,returnPoints=True)
                            perimeter = cv2.arcLength(nearest_contour, True)
                            approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area=cv2.contourArea(nearest_contour)
                            area_r=w*h
                            for item in numbers_detected:
                                if item[1] == 6:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            

                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                                # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                                # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                                if ratio<1.4:
                                    dist=distance((x,y),(Pr_number_x,Pr_number_y))
                                    # print(dist)
                                    if y<(Pr_number_y+10) and middle_y>centery and dist<min_dist and x<Pr_number_x:
                                        min_dist=dist
                                        bboxes=[]
                                        bboxes.append((x,y,w,h,number))
                                    # elif bboxes==[]:
                                    #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                                    #         min_dist=dist
                                    #         bboxes=[]
                                    #         bboxes.append((x,y,w,h,number))

                                    
                        else:
                            condition=False
                    if len(iterations)==0:
                        print('lol')
                    if bboxes:
                        number=bboxes[0][4]
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,7))
                    else:
                        numbers_detected.append(((point[0],point[1],0,0),7))
                        print('Ningun 7 detectado')
                        

                    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                elif angle in [210]:
                    bboxes=get_number(condition,7)
                    # limit_x_max=int(point[0]+70)
                    # limit_x_min=int(point[0]-70)
                    # limit_y_max=int(point[1]+80)
                    # limit_y_min=int(point[1]-80)
                    # # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    # min_dist=float('inf')
                    # iterations=contours.copy()
                    # while condition:
                    #     number,nearest_contour,point_contour=near_contour(iterations,point)
                    #     if number is not None or nearest_contour is not None or point_contour is not None:
                    #         del iterations[number] 
                    #         hull=cv2.convexHull(nearest_contour,returnPoints=True)
                    #         perimeter = cv2.arcLength(nearest_contour, True)
                    #         approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                    #         x, y, w, h = cv2.boundingRect(nearest_contour)
                    #         middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                    #         ratio=w/h
                    #         area=cv2.contourArea(nearest_contour)
                    #         area_r=w*h
                    #         for item in numbers_detected:
                    #             if item[1] == 7:
                    #                 Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                    #             else:
                    #                 Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                    #                 cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)


                    #         if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                    #             # print('Inicio')
                    #             # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                    #             # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                    #             # print(ratio)
                    #             if ratio<1.4:
                    #                 dist=distance((x,y),(Pr_number_x,Pr_number_y))
                    #                 # print(dist)
                    #                 if y<(Pr_number_y+10) and dist<min_dist and x<Pr_number_x:
                    #                     min_dist=dist
                    #                     bboxes=[]
                    #                     bboxes.append((x,y,w,h,number))
                    #                 # elif bboxes==[]:
                    #                 #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                    #                 #         min_dist=dist
                    #                 #         bboxes=[]
                    #                 #         bboxes.append((x,y,w,h,number))

                                    
                    #     else:
                    #         condition=False
                    # if len(iterations)==0:
                    #     print('lol')
                    # if bboxes:
                    #     number=bboxes[0][4]
                    #     del contours[number]
                    #     number_detected=draw_bounding_boxes(draw1,bboxes)
                    #     numbers_detected.append((number_detected,8))
                    # else:
                    #     numbers_detected.append(((point[0],point[1],0,0),8))
                    #     print('Ningun 8 detectado')
                        

                    # cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                
                elif angle in [180]:
                    bboxes=get_number(condition,8)
                        
                    # limit_x_max=int(point[0]+70)
                    # limit_x_min=int(point[0]-70)
                    # limit_y_max=int(point[1]+80)
                    # limit_y_min=int(point[1]-80)
                    # # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    # min_dist=float('inf')
                    # iterations=contours.copy()
                    # while condition:
                    #     number,nearest_contour,point_contour=near_contour(iterations,point)
                    #     if number is not None or nearest_contour is not None or point_contour is not None:
                    #         del iterations[number] 
                    #         hull=cv2.convexHull(nearest_contour,returnPoints=True)
                    #         perimeter = cv2.arcLength(nearest_contour, True)
                    #         approx = cv2.approxPolyDP(nearest_contour, 0.01 * perimeter, True)
                    #         x, y, w, h = cv2.boundingRect(nearest_contour)
                    #         middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                    #         ratio=w/h
                    #         area=cv2.contourArea(nearest_contour)
                    #         area_r=w*h
                    #         for item in numbers_detected:
                    #             if item[1] == 8:
                    #                 Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                    #             else:
                    #                 Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                    #                 cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            

                    #         if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                    #             # print('Inicio')
                    #             # print('Limit_X: '+str(limit_x_min)+'<'+str(middle_x)+'<'+str(limit_x_max))
                    #             # print('Limit_Y: '+str(limit_y_min)+'<'+str(middle_y)+'<'+str(limit_y_max))
                    #             # print(ratio)
                    #             if ratio<1.4:
                    #                 # print(ratio)
                    #                 dist=distance((x,y),(Pr_number_x,Pr_number_y))
                    #                 # print(dist)
                    #                 if y<(Pr_number_y+10) and dist<min_dist and x<Pr_number_x+30:
                    #                     min_dist=dist
                    #                     bboxes=[]
                    #                     bboxes.append((x,y,w,h,number))
                    #                 # elif bboxes==[]:
                    #                 #     if dist<min_dist and dist<65 and ratio<1.1 and y<1.1*centery:
                    #                 #         min_dist=dist
                    #                 #         bboxes=[]
                    #                 #         bboxes.append((x,y,w,h,number))

                                    
                    #     else:
                    #         condition=False
                    # if len(iterations)==0:
                    #     print('lol')
                    # if bboxes:
                    #     number=bboxes[0][4]
                    #     del contours[number]
                    #     number_detected=draw_bounding_boxes(draw1,bboxes)
                    #     numbers_detected.append((number_detected,9))
                    # else:
                    #     numbers_detected.append(((point[0],point[1],0,0),9))
                    #     print('Ningun 9 detectado')
                        

                    # cv2.circle(draw1,tuple(point),2,(0,0,255),-1)

                elif angle in [150]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    # cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area_r=w*h

                            for item in numbers_detected:
                                if item[1] == 9:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            
                            # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1) 
                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max:   
                                dist=distance((x,y),(Pr_number_x,Pr_number_y))
                                if ratio>1.2:
                                    if 300<=area_r<1400:
                                        # bboxes=[]
                                        # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 0, 255), 1)
                                        bboxes.append((x,y,w,h,number))
                                        # cv2.circle(draw1,(middle_x,middle_y),2,(0,0,255),-1)
                                        condition=False
                                else:
                                    # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1)
                                    if 48<=area_r<1300:
                                        bboxes.append((x,y,w,h,number))
                                        # cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                    else:
                                        h=int(h*0.5)
                                        if h>w:
                                            bboxes=[]
                                            bboxes.append((x,y,w,h,number))
                                            # cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                            condition=False  

                                    
                        else:
                            condition=False
                    
                    if len(iterations)==0:
                        print('lol')

                    # Keep track of the closest pair of bounding boxes
                    if len(bboxes)>1:
                        closest_pair = None
                        min_separation = float('inf')
                        closest_bboxes=[]
                        numbers=None
                        min_dist=float('inf')

                        # Iterate over each pair of bounding boxes
                        for i in range(len(bboxes)):
                            centroid1 = calculate_centroid(bboxes[i])
                            x,y,w,h,number_i=bboxes[i]
                            for j in range(i+1, len(bboxes)):
                                x1,y1,w1,h1,number_j=bboxes[j]
                                final_w=w+w1
                                f_ratio=final_w/h
                                dist_y=abs(y-y1)
                                centroid2 = calculate_centroid(bboxes[j])
                                separation = distance(centroid1, centroid2)
                                dist=distance((x1,y1),(Pr_number_x,Pr_number_y))
                                # print(dist_y,dist,f_ratio,bboxes[i],bboxes[j])
                                if dist_y<9 and dist<min_dist and f_ratio<1.7:
                                    # print('final',dist_y,dist,f_ratio)
                                    min_dist=dist
                                    closest_pair = (i, j)
                                    closest_bboxes=[bboxes[i],bboxes[j]]
                                    numbers=(number_i,number_j)
                                elif dist<min_dist and x1>Pr_number_x and y1<Pr_number_y:
                                    closest_bboxes=[bboxes[j]]
                                    numbers=(number_j,number_j)
                                
                                


                        if closest_bboxes:
                            number_detected=draw_bounding_boxes(draw1, closest_bboxes)
                            numbers_detected.append((number_detected,10))

                        if numbers is not None:
                            del contours[numbers[0]]
                            del contours[numbers[1]]
                        else:
                            print("Ningun 10 detectado")

                    elif len(bboxes)==1 and bboxes[0][4] is not None:
                        del contours[bboxes[0][4]]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,10))
                    else:
                        numbers_detected.append(((point[0],point[1],0,0),10))
                        print("Ningun 10 detectado")
               
                elif angle in [120]:
                    limit_x_max=int(point[0]+70)
                    limit_x_min=int(point[0]-70)
                    limit_y_max=int(point[1]+80)
                    limit_y_min=int(point[1]-80)
                    cv2.rectangle(draw1, (limit_x_min, limit_y_min), (limit_x_min +(limit_x_max-limit_x_min), limit_y_min + (limit_y_max-limit_y_min)), (0, 255, 255), 1)
                    min_dist=float('inf')
                    iterations=contours.copy()
                    while condition:
                        number,nearest_contour,point_contour=near_contour(iterations,point)
                        if number is not None or nearest_contour is not None or point_contour is not None:
                            del iterations[number] 
                            x, y, w, h = cv2.boundingRect(nearest_contour)
                            middle_x,middle_y=calculate_centroid((x,y,w,h,number))
                            ratio=w/h
                            area_r=w*h

                            for item in numbers_detected:
                                if item[1] == 10:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                                else:
                                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
                            
                            # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1) 
                            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max:   
                                dist=distance((x,y),(Pr_number_x,Pr_number_y))
                                if ratio>1.2:
                                    if 300<=area_r<1400:
                                        # bboxes=[]
                                        # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 0, 255), 1)
                                        bboxes.append((x,y,w,h,number))
                                        # cv2.circle(draw1,(middle_x,middle_y),2,(0,0,255),-1)
                                        condition=False
                                else:
                                    # cv2.rectangle(draw1, (x, y), (x +w, y + h), (0, 255, 255), 1)
                                    if 48<=area_r<1300:
                                        bboxes.append((x,y,w,h,number))
                                        cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                    else:
                                        h=int(h*0.5)
                                        if h>w:
                                            bboxes=[]
                                            bboxes.append((x,y,w,h,number))
                                            # cv2.rectangle(draw1, (x, y), (x +w, y + h), (255, 0, 0), 1)
                                            condition=False  

                                    
                        else:
                            condition=False
                    
                    if len(iterations)==0:
                        print('lol')

                    # Keep track of the closest pair of bounding boxes
                    if len(bboxes)>1:
                        closest_pair = None
                        min_separation = float('inf')
                        closest_bboxes=[]
                        numbers=None
                        min_dist=float('inf')

                        # Iterate over each pair of bounding boxes
                        for i in range(len(bboxes)):
                            centroid1 = calculate_centroid(bboxes[i])
                            x,y,w,h,number_i=bboxes[i]
                            for j in range(i+1, len(bboxes)):
                                x1,y1,w1,h1,number_j=bboxes[j]
                                final_w=w+w1
                                f_ratio=final_w/h
                                dist_y=abs(y-y1)
                                centroid2 = calculate_centroid(bboxes[j])
                                separation = distance(centroid1, centroid2)
                                dist=distance((x1,y1),(Pr_number_x,Pr_number_y))
                                # print(dist_y,dist,f_ratio,bboxes[i],bboxes[j])
                                if dist_y<15 and dist<100 and f_ratio<1.7 and separation<min_separation:
                                    min_separation=separation
                                    min_dist=dist
                                    # print('final',dist_y,dist,f_ratio)
                                    closest_pair = (i, j)
                                    closest_bboxes=[bboxes[i],bboxes[j]]
                                    numbers=(number_i,number_j)
                                elif dist<min_dist and x1>Pr_number_x and y1<Pr_number_y:
                                    closest_bboxes=[bboxes[j]]
                                    numbers=(number_j,number_j)
                                
                                


                        if closest_bboxes:
                            number_detected=draw_bounding_boxes(draw1, closest_bboxes)
                            numbers_detected.append((number_detected,11))

                        if numbers is not None:
                            del contours[numbers[0]]
                            del contours[numbers[1]]
                        else:
                            print("Ningun 11 detectado")

                    elif len(bboxes)==1 and bboxes[0][4] is not None:
                        del contours[bboxes[0][4]]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,11))
                    else:
                        numbers_detected.append(((point[0],point[1],0,0),11))
                        print("Ningun 11 detectado")
                        

                    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
                # if numbers_detected:
                #     print('si')
                # else:
                #     print('no')
            cv2.drawContours(draw1,contours,-1,(0,0,255),1)
            # print(numbers_detected)

        print(numbers_detected)


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