from pdf2image import convert_from_path
import argparse
import os
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--path_to_PDF',type=str,required=True)
parser.add_argument('--path_to_saved_images',type=str,required=True)
args=parser.parse_args()

folder_exist=os.path.exists(args.path_to_saved_images)
if not folder_exist:
    os.makedirs(args.path_to_saved_images)
    print("A new directory to save the images has been created!")
for file in os.listdir(args.path_to_PDF):
    if file.endswith(".pdf"):
        file_name,extension=os.path.splitext(file)   
        test=convert_from_path(args.path_to_PDF+'/'+file,output_folder=args.path_to_saved_images,fmt='png',first_page=1,last_page=1,output_file=file_name)