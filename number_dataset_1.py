import numpy as np
import cv2
import argparse
import tensorflow as tf
import os
import time
import math

# cv2.namedWindow('MNIST',cv2.WINDOW_NORMAL)
# cv2.namedWindow('Artificial',cv2.WINDOW_NORMAL)

parser=argparse.ArgumentParser()
parser.add_argument('--path_to_save_images',type=str,required=True)
parser.add_argument('--path_to_save_labels',type=str,required=True)
args=parser.parse_args()

folder_exist=os.path.exists(args.path_to_save_images)
if not folder_exist:
    os.makedirs(args.path_to_save_images)
    print("A new directory to save the IMAGES has been created!")

folder_exist=os.path.exists(args.path_to_save_labels)
if not folder_exist:
    os.makedirs(args.path_to_save_labels)
    print("A new directory to save the LABELS has been created!")

def generate_synthetic_data(j=0,resize=False,resize_width=28,resize_height=28):
    Artificial_image=np.zeros((420,450),dtype=np.uint8)
    grid = np.zeros((Artificial_image.shape[0] // resize_height, Artificial_image.shape[1] // resize_width), dtype=bool)

    clock_positions = {}
    center_x = Artificial_image.shape[1] // 2
    center_y = Artificial_image.shape[0] // 2
    for number in range(1, 13):
        angle = math.radians(-(number -3 ) * 30)  # Adjusting the angle for clockwise arrangement, starting from 1
        radius = 60  # Adjust the radius as needed
        x_pos = center_x + int(radius * math.cos(angle))
        y_pos = center_y - int(radius * math.sin(angle))  # Subtracting for clockwise direction
        clock_positions[number] = (x_pos, y_pos)


    mnist=tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    added_labels=set()
    added_numbers=set()

    digit_indices_train={}

    for digit in range(10):
        indices_train=np.where(y_train==digit)[0]
        digit_indices_train[digit]=indices_train

    min_length = min(len(indices) for indices in digit_indices_train.values())

    # print(min_length)

    for i in range(min_length):

        with open(args.path_to_save_labels+'/'+'Artificial_image_'+str(j)+'.txt', 'w') as file:
            pass  # This does nothing but effectively clears the file
        
        unit=0
        for number in range(1,len(digit_indices_train)+3):
            if number>9:
                idx1=digit_indices_train[1][i]
                idx2=digit_indices_train[unit][i]
                image1=x_train[idx1]
                label1=y_train[idx1]
                image2=x_train[idx2]
                label2=y_train[idx2]
                image1_resized = cv2.resize(image1, (14, 28))
                image2_resized = cv2.resize(image2, (14, 28))
                image = np.concatenate((image1_resized, image2_resized), axis=1)
                label=str(label1)+str(label2)
                unit+=1

            else:           
                idx=digit_indices_train[number][i]
                image=x_train[idx]
                label=y_train[idx]

            if resize:
                image = cv2.resize(image, (resize_width, resize_height))
        
            while True:
                row = np.random.randint(0, grid.shape[0])
                col = np.random.randint(0, grid.shape[1])

                # Check if neighboring cells are empty
                if not np.any(grid[max(0, row-1):min(row+2, grid.shape[0]),
                                    max(0, col-1):min(col+2, grid.shape[1])]):
                    break

            # Update grid
            grid[row, col] = True
            
            x_pos = col * resize_width
            y_pos = row * resize_height
            x_pos, y_pos = clock_positions[number]


            Artificial_image[y_pos:y_pos+resize_height,x_pos:x_pos+resize_width]=image

            x_center=x_pos+0.5*resize_width
            y_center=y_pos+0.5*resize_height

            x_center_normalized = x_center / Artificial_image.shape[1]
            y_center_normalized = y_center / Artificial_image.shape[0]

            width_normalized=resize_width/Artificial_image.shape[1]
            heigh_normalized=resize_height/Artificial_image.shape[0]

            data=f"{label} {x_center_normalized} {y_center_normalized} {width_normalized} {heigh_normalized} "
            with open(args.path_to_save_labels+'/'+'Artificial_image_'+str(j)+'.txt','a') as file:
                file.write(data+'\n')

            # print('label is: ',label)
            # cv2.imshow('MNIST',noise)
            # cv2.imshow('Artificial',Artificial_image)
            # cv2.waitKey(0)
            
        noise = np.random.randint(0, 150, Artificial_image.shape, dtype=np.uint8)
        noisy_mask = cv2.add(Artificial_image, noise)
        # cv2.imshow('lol',noisy_mask)
        # cv2.waitKey(0)

        cv2.imwrite(args.path_to_save_images+'/'+'Artificial_image_'+str(j)+'.png',Artificial_image)
        Artificial_image=np.zeros((420,450),dtype=np.uint8)
        added_labels=set()
        grid = np.zeros((Artificial_image.shape[0] // resize_height, Artificial_image.shape[1] // resize_width), dtype=bool)
        j+=1
        print(j)
    return j

last_point=generate_synthetic_data(j=0,resize=True,resize_width=28,resize_height=28)
# last_point=generate_synthetic_data(j=last_point,resize=True,resize_width=int(28*1.25),resize_height=int(28*1.25))
# last_point=generate_synthetic_data(j=last_point,resize=True,resize_width=int(28*1.5),resize_height=int(28*1.5))
# last_point=generate_synthetic_data(j=last_point,resize=True,resize_width=int(28*1.75),resize_height=int(28*1.75)) 
cv2.destroyAllWindows()
