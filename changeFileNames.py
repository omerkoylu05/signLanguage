import os

# Specify the directory
dir_path = './dataset/Ue'

# Specify the prefix
prefix = 'Ue'

# Iterate over all files in the directory
for filename in os.listdir(dir_path):
    # Create the new file name by replacing the first letter with the prefix
    new_filename = prefix + filename[1:]

    # Rename the file
    os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))
