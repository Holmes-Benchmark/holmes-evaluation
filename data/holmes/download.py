import os
import gdown
from os.path import exists

current_directory = os.path.split(os.getcwd())[1]

url = "https://drive.google.com/file/d/1rkTka-k5dE2GqAXJxEcgI2FJGCaQvDVd/view?usp=sharing"

output_file = "holmes-free.zip"

if current_directory != "holmes":
    print("This is the wrong directory. Please run this script in the folder data/flash-holmes")
else:
    gdown.download(url=url, output=output_file, fuzzy=True)

    if not exists("holmes-free.zip"):
        print("Download failed. Please try again.")
    else:
        os.system("unzip holmes-free.zip")