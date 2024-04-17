from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import random

# Load a model
model = YOLO('/home/enchi/Documentos/Biggest_Dataset.pt')  # pretrained YOLOv8n model

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
    bboxes=[]
    limit_x_max=int(point[0]+50)
    limit_x_min=int(point[0]-80)
    limit_y_max=int(point[1]+60)
    limit_y_min=int(point[1]-70)
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
                if item[1] == reference:
                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=item[0]
                    Pr_area_r=Pr_number_w*Pr_number_h
                    Pr_number_middle_x,Pr_number_middle_y=calculate_centroid((Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h,item[1]))

                else:
                    Pr_number_x,Pr_number_y,Pr_number_w,Pr_number_h=(centerx+6,0,0,0)
                    cv2.circle(draw1,(centerx,centery),2,(0,0,255),-1)
            
            if limit_x_min<middle_x<limit_x_max and limit_y_min<middle_y<limit_y_max and area_r>69: 
                if ratio<1.5:
                    dist=distance((x,y),(Pr_number_x,Pr_number_y))
                    if reference in [6,7,8]:
                        if middle_y<(Pr_number_middle_y+10) and dist<min_dist and middle_x<Pr_number_middle_x+20 and 100<area_r<2000:
                            min_dist=dist
                            bboxes=[]
                            bboxes.append((x,y,w,h,number))
                    elif reference in [3,4,5]:
                        if middle_y>(Pr_number_middle_y-10) and dist<min_dist and Pr_area_r*0.3<area_r<2000 and x<Pr_number_x+10:
                            min_dist=dist
                            bboxes=[]
                            bboxes.append((x,y,w,h,number))
                            
                    elif reference in [0,1,2]:
                        if middle_y>Pr_number_middle_y-10 and dist<min_dist and middle_x>=Pr_number_middle_x-10 and 100<area_r<2000:
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
        numbers_detected.append((number_detected,reference+1))
    else:
        numbers_detected.append(((point[0],point[1],0,0),reference+1))
        print('Ningun '+str(reference+1)+' detectado')

    cv2.circle(draw1,tuple(point),2,(0,0,255),-1)
    return bboxes

def get_2_numbers(condition,reference):
    bboxes=[]
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
                if item[1] == reference:
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
            numbers_detected.append((number_detected,reference+1))

        if numbers is not None:
            del contours[numbers[0]]
            del contours[numbers[1]]
        else:
            print("Ningun"+str(reference+1) +"detectado")

    elif len(bboxes)==1 and bboxes[0][4] is not None:
        del contours[bboxes[0][4]]
        number_detected=draw_bounding_boxes(draw1,bboxes)
        numbers_detected.append((number_detected,reference+1))
    else:
        numbers_detected.append(((point[0],point[1],0,0),reference+1))
        print("Ningun"+str(reference+1) +"detectado")
    
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
        mask = np.zeros_like(source)
        bboxes_img=np.zeros_like(gray)
    
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
                    limit_x_max=int(point[0]+40)
                    limit_x_min=int(point[0]-40)
                    limit_y_max=int(point[1]+60)
                    limit_y_min=int(point[1]-70)
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
                            numbers_detected.append((number_detected,0))

                        if numbers is not None:
                            del contours[numbers[0]]
                            del contours[numbers[1]]
                        else:
                            print("Ningun 12 detectado")
                            numbers_detected.append(((point[0],point[1],0,0),0))
            
                    elif len(bboxes)==1 and number is not None:
                        del contours[number]
                        number_detected=draw_bounding_boxes(draw1,bboxes)
                        numbers_detected.append((number_detected,0))
                    else:
                        print("Ningun 12 detectado")
                        numbers_detected.append(((point[0],point[1],0,0),0))
                                
                elif angle in [60]:
                    bboxes=get_number(condition,0)
                elif angle in [30]:
                    bboxes=get_number(condition,1)
                elif angle in [0]:
                    bboxes=get_number(condition,2)
                elif angle in [330]:
                    bboxes=get_number(condition,3)
                elif angle in [300]:
                    bboxes=get_number(condition,4)
                elif angle in [270]:
                    bboxes=get_number(condition,5)
                elif angle in [240]:
                    bboxes=get_number(condition,6)
                elif angle in [210]:
                    bboxes=get_number(condition,7)
                elif angle in [180]:
                    bboxes=get_number(condition,8)
                elif angle in [150]:
                    get_2_numbers(condition,9)                   
                elif angle in [120]:
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
 
            cv2.drawContours(draw1,contours,-1,(0,0,255),1)
        # print(numbers_detected)
        class_names=[]
        labels=[]
        print('Initiatin predictions')
        for bbox,label in numbers_detected:
            x,y,w,h=bbox
            mask = np.zeros_like(source)
            mask[y:y+h,x:x+w]=source[y:y+h,x:x+w] 
            if label==0:
                label=12     
            # cv2.imshow('mask',mask)              
            results = model(mask, conf=0.6, iou=0.7,max_det=200,imgsz=448,verbose=False)
            xyxys = []
            confidences = []
            class_ids = []
            if w !=0 and h!=0:
                for result in results:
                    cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), 255, -1)
                    boxes = result.boxes.cpu().numpy()
                    xyxys.extend(boxes.xyxy)  # Use extend to append multiple values
                    confidences.extend(boxes.conf)
                    class_ids.extend(boxes.cls)
                    if xyxys==[]:
                        print('something detected but not preddicted')
                        cv2.rectangle(draw, (x, y), (x+w, y+h), (255, 0,0), 1)
                        text = f'{label}: {0:.2f}'
                        cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    for i, xyxy in enumerate(xyxys):
                        x1, y1, x2, y2 = map(int, xyxy)
                        class_name = class_ids[i]
                        labels.append(label)
                        class_names.append(class_name)
                        if class_name==label:
                            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255,0), 1)
                            print('Correct preddiction')
                            confidence = confidences[i]
                            text = f'{class_name}: {confidence:.2f}'
                            cv2.putText(draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        elif class_name==label+1 or class_name==label-1:
                            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 165,255), 1)
                            confidence = confidences[i]
                            text = f'{class_name}: {confidence:.2f}'
                            cv2.putText(draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                            print("Error in arrangement")
                        else:
                            cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 0,255), 1)
                            text = f'{label}: {0:.2f}'
                            cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            print("Need to check")
                       
        result_mask = cv2.bitwise_and(clock_contour, bboxes_img)
        overlap_percentage = np.sum(result_mask) / np.sum(bboxes_img)
        print(overlap_percentage)
       
        #------------------------------------------------------------------------------------------
        # cv2.imshow('draw', draw)
        # cv2.namedWindow('source',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('source', 800, 600)  # Set the width and height of the window
        # cv2.imshow('source',draw1)
        # cv2.imshow('mask',mask)
        # cv2.imshow('gray',gray)
        # cv2.imshow('test',test)
        cv2.imshow('source',source)
        cv2.imshow('circle',clock_contour)
        cv2.imshow('draw', draw)
        # cv2.imshow('bboxes',bboxes_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()