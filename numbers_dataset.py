import numpy as np
import cv2
import argparse
import tensorflow as tf
import os
import time

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

Artificial_image=np.zeros((420,450),dtype=np.uint8)

grid = np.zeros((Artificial_image.shape[0] // 28, Artificial_image.shape[1] // 28), dtype=bool)

mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


shuffled_indices = np.arange(len(x_train))
np.random.shuffle(shuffled_indices)

added_labels=set()
added_numbers=set()

digit_indices_train={}

for digit in range(10):
    indices_train=np.where(y_train==digit)[0]
    digit_indices_train[digit]=indices_train

# for digit in range(10):
#     indices_test=np.where(y_test==digit)[0]
#     digit_indices_test[digit]=indices_test

# print(digit_indices_test)



min_length = min(len(indices) for indices in digit_indices_train.values())

for i in range(min_length):

    with open(args.path_to_save_labels+'/'+'Artificial_image_'+str(i)+'.txt', 'w') as file:
        pass  # This does nothing but effectively clears the file

    for number in range(len(digit_indices_train)):
        idx=digit_indices_train[number][i]
        image=x_train[idx]
        label=y_train[idx]
       
        while True:
            row = np.random.randint(0, grid.shape[0])
            col = np.random.randint(0, grid.shape[1])

            # Check if neighboring cells are empty
            if not np.any(grid[max(0, row-1):min(row+2, grid.shape[0]),
                                max(0, col-1):min(col+2, grid.shape[1])]):
                break

         # Update grid
        grid[row, col] = True
        
        x_pos = col * 28
        y_pos = row * 28

        Artificial_image[y_pos:y_pos+28,x_pos:x_pos+28]=image

        x_center=x_pos+0.5*28
        y_center=y_pos+0.5*28

        x_center_normalized = x_center / Artificial_image.shape[1]
        y_center_normalized = y_center / Artificial_image.shape[0]

        width_normalized=28/Artificial_image.shape[1]
        heigh_normalized=28/Artificial_image.shape[0]

        data=f"{label} {x_center_normalized} {y_center_normalized} {width_normalized} {heigh_normalized} "
        with open(args.path_to_save_labels+'/'+'Artificial_image_'+str(i)+'.txt','a') as file:
            file.write(data+'\n')

        # print('label is: ',label)
        # cv2.imshow('MNIST',image)
        # cv2.imshow('Artificial',Artificial_image)
        # cv2.waitKey(0)

    
        cv2.imwrite(args.path_to_save_images+'/'+'Artificial_image_'+str(i)+'.png',Artificial_image)
        Artificial_image=np.zeros((420,450),dtype=np.uint8)
        added_labels=set()
        grid = np.zeros((Artificial_image.shape[0] // 28, Artificial_image.shape[1] // 28), dtype=bool)
    
print(i)

    
cv2.destroyAllWindows()
