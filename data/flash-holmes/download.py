import os
import gdown
from os.path import exists

current_directory = os.path.split(os.getcwd())[1]

url = "https://drive.google.com/file/d/1dajhNPzBP4QzDT4bfYSp_mPHToxJSO_L/view?usp=sharing"

output_file = "flash-holmes.zip"

if current_directory != "flash-holmes":
    print("This is the wrong directory. Please run this script in the folder data/flash-holmes")
else:
    gdown.download(url=url, output=output_file, fuzzy=True)

    if not exists("flash-holmes.zip"):
        print("Download failed. Please try again.")
    else:
        os.system("unzip flash-holmes.zip")