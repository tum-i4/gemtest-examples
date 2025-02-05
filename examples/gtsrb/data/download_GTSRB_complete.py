import os
import urllib.request
import zipfile
from pathlib import Path


def download_data():
    if Path("GTSRB_Complete").is_dir():
        print("GTSRB_Complete folder already exists. No action taken ...")

    metadata_dir = Path("GTSRB_Complete/GTSRB_Metadata")
    image_dir = Path("GTSRB_Complete/GTSRB_Images")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Metadata ...")
    urllib.request.urlretrieve(
        url="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
            "GTSRB_Final_Test_GT.zip",
        filename="GTSRB_Complete/GTSRB_Metadata/GTSRB_Final_Test_GT.zip")

    with zipfile.ZipFile("GTSRB_Complete/GTSRB_Metadata/GTSRB_Final_Test_GT.zip",
                         'r') as zip_ref:
        zip_ref.extractall("GTSRB_Complete/GTSRB_Metadata")

    os.remove("GTSRB_Complete/GTSRB_Metadata/GTSRB_Final_Test_GT.zip")
    print("Metadata download completed.")

    print("Downloading Images ...")
    urllib.request.urlretrieve(
        url="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
            "GTSRB_Final_Test_Images.zip",
        filename="GTSRB_Complete/GTSRB_Images/GTSRB_Complete.zip")

    with zipfile.ZipFile("GTSRB_Complete/GTSRB_Images/GTSRB_Complete.zip", 'r') as zip_ref:
        zip_ref.extractall("GTSRB_Complete/GTSRB_Images")

    os.remove("GTSRB_Complete/GTSRB_Images/GTSRB/Final_Test/Images/GT-final_test.test.csv")
    os.remove("GTSRB_Complete/GTSRB_Images/GTSRB_Complete.zip")
    print("Image download completed.")


download_data()
