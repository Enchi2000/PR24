import os
import argparse
import csv
import pandas as pd
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--path_to_folder',type=str,required=True)
parser.add_argument('--path_to_save_csv',type=str,required=True)
args=parser.parse_args()

def evaluate(value,name):
    while True:
        try:
            value=int(value)
            if value in [0, 1]:
                break
            else:
                print("Invalid input. Please enter either 0 or 1.")
                value = input(name + ': ')
        except ValueError:
            print("Invalid input. Enter a valid number")
            value=input(name+': ')
    return value

cv2.namedWindow('Img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Img',1920,1080)

data=[]

folder_exist=os.path.exists(args.path_to_save_csv)
if not folder_exist:
    os.makedirs(args.path_to_save_csv)
    print("A new directory to save the csv has been created!")

columns = ['image_name','classes','numbers','contour','clock hands']

df=pd.DataFrame(columns=columns)

for folder in os.listdir(args.path_to_folder):
    folder_path=os.path.join(args.path_to_folder,folder)
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            file_name,extension=os.path.splitext(file)
            img=cv2.imread(folder_path+'/'+file)
            cv2.imshow('Img',img)
            cv2.waitKey(1)  
            Countour=input("Contour: ")
            Countour=evaluate(Countour,'Countour')
            Numbers=input("Numbers: ")
            Numbers=evaluate(Numbers,'Numbers')
            Clock_hands=input("Clock hands: ")
            Clock_hands=evaluate(Clock_hands,'Clock hands')
            data=pd.DataFrame([[file_name+extension,folder,Numbers,Countour,Clock_hands]],columns=columns)
            df=pd.concat([df,data],axis=0)
            print(df)

cv2.destroyAllWindows()


file_name = 'labels.csv'
print(args.path_to_save_csv+file_name)

df.to_csv(args.path_to_save_csv+'/'+file_name, index=False)

