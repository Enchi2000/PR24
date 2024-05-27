"""
Description:
This script converts the first page of each PDF file in a specified directory into a PNG image. It utilizes the pdf2image library to perform the conversion.

Usage:
- Run the script with `python script_name.py`.
- Specify the path to the directory containing PDF files using the `--path_to_PDF` argument.
- Specify the path to save the converted images using the `--path_to_saved_images` argument.

Example:
python script_name.py --path_to_PDF /path/to/PDF/files --path_to_saved_images /path/to/save/images
"""
# Import the necessary functions and modules.
# convert_from_path: Function to convert PDF pages to images.
# argparse: Module for parsing command-line arguments.
# os: Module for interacting with the operating system.
# Image: Module from PIL (Python Imaging Library) for image processing.
from pdf2image import convert_from_path
import argparse
import os
from PIL import Image

# Create an ArgumentParser object to handle command-line arguments.
parser = argparse.ArgumentParser()

# Add an argument for the path to the PDF files.
parser.add_argument('--path_to_PDF', type=str, required=True, help='Path to the directory containing PDF files.')

# Add an argument for the path to save the converted images.
parser.add_argument('--path_to_saved_images', type=str, required=True, help='Path to the directory to save the converted images.')

# Parse the command-line arguments and store them in the 'args' variable.
args = parser.parse_args()

# Check if the directory to save the images exists.
folder_exist = os.path.exists(args.path_to_saved_images)

# If the directory does not exist, create it.
if not folder_exist:
    os.makedirs(args.path_to_saved_images)
    print("A new directory to save the images has been created!")

# Loop through all the files in the specified PDF directory.
for file in os.listdir(args.path_to_PDF):
    # Check if the file has a .pdf extension.
    if file.endswith(".pdf"):
        # Split the file name and its extension.
        file_name, extension = os.path.splitext(file)

        # Convert the first page of the PDF to a PNG image.
        # args.path_to_PDF + '/' + file: Full path to the PDF file.
        # output_folder: Directory to save the output images.
        # fmt='png': Format of the output image.
        # first_page=1, last_page=1: Convert only the first page.
        # output_file=file_name: Use the PDF file name for the output image.
        convert_from_path(
            args.path_to_PDF + '/' + file,
            output_folder=args.path_to_saved_images,
            fmt='png',
            first_page=1,
            last_page=1,
            output_file=file_name
        )
