#Importar librerias
import os
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
#-----------------------------------------
#Definir argumentos de la terminal
parser=argparse.ArgumentParser()
parser.add_argument('--path_to_roi',type=str,required=True)
parser.add_argument('--path_to_save_contour',type=str,required=True)
args=parser.parse_args()
#-----------------------------------------

#Ventanas para visualizar imagenes
cv2.namedWindow('Img',cv2.WINDOW_NORMAL)
cv2.namedWindow('contour',cv2.WINDOW_NORMAL)
cv2.namedWindow('rest',cv2.WINDOW_NORMAL)
cv2.namedWindow('numbers',cv2.WINDOW_NORMAL)
cv2.namedWindow('hand_clock',cv2.WINDOW_NORMAL)
#-------------------------------------------

#Inicializacion de variables
detected_inner_contours=[]
detected_outer_contours=[]
outer_contours=[]
hull_detected=[]
inner_contours=[]
final_shape=[]
direction_change_points = []
points=[]


min_distance = float('inf')

#--------------------------------------------

#Crea un directorio para guardar las imagenes en caso de que no exista
folder_exist=os.path.exists(args.path_to_save_contour)
if not folder_exist:
    os.makedirs(args.path_to_save_contour)
    print("A new directory to save the CONTOUR images has been created!")
#-------------------------------------------------------------------------

#Read files in ROI directoru
for file in os.listdir(args.path_to_roi):
    #Check if files end with .PNG
    if file.endswith(".png"):
        #Split file name into file,extension
        file_name,extension=os.path.splitext(file)
        #Read image
        img=cv2.imread(args.path_to_roi+'/'+file)
        #Get the height and width of image
        height, width, _ = img.shape
        #Transforms image into grayscale
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #Create mask for future bitwise operations
        mask=np.zeros_like(img_gray)
        mask1=np.zeros_like(img_gray)
        mask2=np.zeros_like(img_gray)
        mask3=np.zeros_like(img_gray)
        mask4=np.zeros_like(img_gray)
        line_image = np.zeros_like(img_gray)

        #----------------------------------------

        #Apply Gaussian Filter to grayscale image
        blurred=cv2.GaussianBlur(img_gray,(3,3),0)
        #Apply laplacian filter to blur image
        laplacian=cv2.Laplacian(blurred,cv2.CV_16S)
        #Make the image sharper
        sharp_image = cv2.convertScaleAbs(laplacian-blurred)
        #Apply binarization to image
        ret,img_threshold=cv2.threshold(sharp_image,229,251,cv2.THRESH_BINARY_INV)

        #Find contours in img
        contours,hierarchy=cv2.findContours(img_threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #For every contour in contours find outer and inner contours based in hierarchy
        for i in range(len(contours)):
            if hierarchy[0][i][3]==-1:
                outer_contours.append(contours[i])

            if hierarchy[0][i][3]!=-1:
                inner_contours.append(contours[i])
        #For every inner contour find the area
        for contour in inner_contours:
            area=cv2.contourArea(contour)
            #If area is greater than 1000, add to the list
            if area>1000:
                detected_inner_contours.append(contour)
        #For every outer contour find the area
        for contour in outer_contours:
            area=cv2.contourArea(contour)
            #If area is greater than 1000, find the convex hull and add to the list
            if area>1000:
                hull=cv2.convexHull(contour,returnPoints=True)
                detected_outer_contours.append(contour)
                hull_detected.append(hull)

        #Define kernel for morphological operations
        kernel=np.ones((5,5),np.uint8)

        #If detected inner and outer contour
        if detected_inner_contours and detected_outer_contours:
            #Draw inner contours in mask
            cv2.drawContours(mask,detected_inner_contours,-1,(255,255,255),thickness=cv2.FILLED)
            #Draw outer contours in mask
            cv2.drawContours(mask1,detected_outer_contours,-1,(255,255,255),thickness=cv2.FILLED)
            #Apply bitwise not to outer contours to detect outside the region
            mask1=cv2.bitwise_not(mask1)
            #Make white area bigger of inner and outside the outer contour
            inner_contour_dilated=cv2.dilate(mask,kernel,iterations=3)
            outer_contour_dilated=cv2.dilate(mask1,kernel,iterations=3)
            #Where areas meet it should be the contour of the image
            result=cv2.bitwise_and(outer_contour_dilated,inner_contour_dilated)

        #Only detect outer contours
        else:
            #Draw the hull contour of outer contours
            result=cv2.drawContours(mask1,hull_detected,-1,(255,255,255),thickness=cv2.FILLED)
            #Make hull bigger for better detection of contour
            result=cv2.dilate(result,kernel,iterations=1)
            #Make the hull smaller
            adjust=cv2.erode(result,kernel,iterations=4)
            #Substract the adjust to result to obtain only the contour
            result=result-adjust

        #Close any open contour
        result=cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        contours,hierarchy=cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for i,contour in enumerate(contours):
            if hierarchy[0][i][3]==-1:
                hull=cv2.convexHull(contour,returnPoints=True)
                final_shape.append(hull)

        cv2.drawContours(mask2,final_shape,-1,(255,255,255),thickness=cv2.FILLED)
        adjust=cv2.erode(mask2,kernel,iterations=4)
        mask2=mask2-adjust
        other=cv2.bitwise_not(mask2)

        final_contour=cv2.bitwise_and(img,img,mask=mask2)
        rest=cv2.bitwise_and(img,img,mask=other)

        #Make 3 channel masks
        three_channel_mask=cv2.merge([mask2]*3)
        three_channel_mask1=cv2.merge([other]*3) 

        rest=rest+three_channel_mask  
        final_contour=final_contour+three_channel_mask1 

        #Get the moments from mask
        M=cv2.moments(mask2)
        #Calculate the center of contour
        centerx = int(M['m10'] / M['m00'])
        centery = int(M['m01'] / M['m00'])
        center = (centerx, centery)
        #Draw a circle in the image
        # cv2.circle(img, center, 2, (0, 255, 255), -1)

        rest_gray=cv2.bitwise_and(sharp_image,sharp_image,mask=other)
        rest_gray=rest_gray+mask2
        ret,rest_gray_th=cv2.threshold(rest_gray,229,251,cv2.THRESH_BINARY_INV)
        #--------------------------------------------------------------------
        #Close any gaps in clock hands
        kernel=np.ones((3,3),np.uint8)
        adjust2=cv2.erode(adjust,kernel,iterations=30)
        numbers=cv2.bitwise_and(img,img,mask=adjust2)
        closing = cv2.morphologyEx(rest_gray_th, cv2.MORPH_CLOSE, kernel)

        contours,hierarchy=cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            # Calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                contour_centroid_x = int(M["m10"] / M["m00"])
                contour_centroid_y = int(M["m01"] / M["m00"])
                # Calculate distance between contour centroid and image centroid
                distance = np.sqrt((centerx - contour_centroid_x)**2 + (centery - contour_centroid_y)**2)
                 # Update nearest contour if distance is smaller
                if distance < min_distance and perimeter>100:
                    min_distance = distance
                    nearest_contour = contour

        cv2.drawContours(mask4,[nearest_contour],-1,(255,255,255),thickness=cv2.FILLED)
        mask4=cv2.bitwise_and(mask4,mask4,mask=adjust2)
        hand_clock=cv2.bitwise_and(img,img,mask=mask4)

        mask5=mask2+mask4
        numbers=cv2.bitwise_not(mask5)
        numbers=cv2.bitwise_and(img,img,mask=numbers)
        three_channel_mask2=cv2.merge([mask5]*3)
        numbers=numbers+three_channel_mask2

        contours, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            test=cv2.convexHull(contour)
            lowest_point = (0,0)
            threshold_distance = 40
            for point in test:
                x, y = point[0]
                points.append((x,y))
                distance_to_centroid = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if y > lowest_point[1] and distance_to_centroid < threshold_distance:
                    lowest_point = (x, y)
                # cv2.circle(img, tuple(point[0]), 3, (0, 255, 0), -1)  # Draw a filled circle at each point
        if lowest_point==(0,0):
            lowest_point = center
        
        if points is None or len(points) == 0:
            print("Error: No points data provided.")
        else:
            
            num_clusters=5
            points_array=np.array(points)
            points_for_clustering = np.float32(points)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters,init='k-means++',max_iter=300,tol=1e-4,random_state=42)
            kmeans.fit(points_for_clustering)

            # Get the cluster centers
            cluster_centers = kmeans.cluster_centers_

            img_with_clusters = img.copy()
            for cluster_center in cluster_centers:
                cv2.circle(img_with_clusters, (int(cluster_center[0]), int(cluster_center[1])), 2, (0, 0, 255), -1)
                cv2.line(img_with_clusters, lowest_point, (int(cluster_center[0]), int(cluster_center[1])), (0, 255, 0), 2)
                cv2.circle(img_with_clusters, lowest_point, 2, (255, 0, 0), -1)
                cv2.circle(img_with_clusters, center, 2, (255, 255, 0), -1)
            
            cv2.imshow('Cluster Centers', img_with_clusters)

        #---------------------------------------------------------------------
        # Show windows with images
        cv2.imshow('Img',img)
        cv2.imshow('contour',final_contour)
        cv2.imshow('rest',rest)
        cv2.imshow('numbers',numbers)
        cv2.imshow('hand_clock',mask4)

        #----------------------
        
        #Save the image into the directory
        # cv2.imwrite(args.path_to_save_contour+'/'+file_name+'.png',rest_gray_th)

        #Wait unti key is pressed
        cv2.waitKey(0)

        #Reset all variables
        detected_outer_contours=[]
        detected_inner_contours=[]
        outer_contours=[]
        hull_detected=[]
        inner_contours=[]
        final_shape=[]
        direction_change_points = []
        points=[]
        min_distance = float('inf') 

#Close all windows
cv2.destroyAllWindows()

