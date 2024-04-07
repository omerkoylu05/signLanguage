import os
import shutil

# Specify the source folder where all images are currently stored
source_folder = './dataset'

# Loop through all files in the source folder
for file_name in os.listdir(source_folder):
    # Get the first letter of each file name (i.e., the letter of the alphabet the image represents)
    first_letter = file_name[0]

    # Create a new folder path
    new_folder_path = os.path.join(source_folder, first_letter)

    # If this folder doesn't already exist, create it
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Move the file to the new folder
    shutil.move(os.path.join(source_folder, file_name), new_folder_path)
