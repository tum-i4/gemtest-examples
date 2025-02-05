"""
This module will download the SVHN test dataset and extract digits under the SVHN_Complete folder
"""

import os
import tarfile
from pathlib import Path

import h5py
import requests
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
test_filename = 'test.tar.gz'
test_folder = f"{path}/SVHN_Complete"


def download_test_dataset():
    if Path("SVHN_Complete").is_dir():
        print("SVHN_Complete folder already exists. No action taken ...")
        return

    image_dir = Path(test_folder)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_file = os.path.join(image_dir, test_filename)

    print("Downloading images ...")

    url = f"http://ufldl.stanford.edu/housenumbers/{test_filename}"

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Open the file in binary write mode and write the content
    with open(image_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    with tarfile.open(image_file, "r:gz") as tar:
        tar.extractall(path=image_dir)
        print(f"File extracted to {image_dir}")


def read_string(ref, f):
    # Dereference the reference and read the string data
    str_data = f[ref]
    str_data = str_data[()]
    string = ''.join(chr(i[0]) for i in str_data)
    return string


def read_bounding_boxes(box_ref, f):
    bbox_metadata = {}
    keys = ['height', 'left', 'top', 'width', 'label']

    for key in keys:
        attr = f[box_ref][key]
        if len(attr) > 1:
            bbox_metadata[key] = [int(f[a.item()][()][0][0]) for a in attr]
        else:
            bbox_metadata[key] = [int(attr[()][0][0])]

    return bbox_metadata


def read_digit_struct(file_path):
    with h5py.File(file_path, 'r') as f:
        digit_struct_group = f['digitStruct']
        names = digit_struct_group['name']
        bboxes = digit_struct_group['bbox']

        all_img_metadata = []
        for i in range(len(names)):
            # img_name = ''.join(chr(c) for c in f[names[i][0]])
            img_name_ref = names[i][0]
            img_name = read_string(img_name_ref, f)

            bbox_ref = bboxes[i].item()
            bbox_metadata = read_bounding_boxes(bbox_ref, f)

            all_img_metadata.append((img_name, bbox_metadata))
        return all_img_metadata


def extract_images():
    print("Extracting digits ...")
    downloaded_dir = Path(test_folder)
    downloaded_file = os.path.join(downloaded_dir, 'test')

    digit_struct_path = os.path.join(downloaded_file, 'digitStruct.mat')
    all_img_metadata = read_digit_struct(digit_struct_path)

    svhn_meta_folder = Path(os.path.join(downloaded_dir, "SVHN_Metadata"))
    svhn_images_folder = Path(os.path.join(downloaded_dir, "SVHN_Images/SVHN/Final_Test"))
    svhn_meta_folder.mkdir(parents=True, exist_ok=True)
    svhn_images_folder.mkdir(parents=True, exist_ok=True)

    for img_name, bboxes in all_img_metadata:
        img_path = os.path.join(downloaded_file, img_name)
        img = Image.open(img_path)

        # File to write labels
        labels_file_path = os.path.join(svhn_meta_folder, 'labels.txt')
        with open(labels_file_path, 'a') as labels_file:
            for i, (label, left, top, height, width) in enumerate(
                    zip(bboxes['label'], bboxes['left'], bboxes['top'], bboxes['height'], bboxes['width'])):
                # Extract and save each digit as a PNG
                digit = img.crop((left, top, left + width, top + height))
                digit_file_name = f"{img_name[:-4]}_digit_{i}.png"
                digit_path = os.path.join(svhn_images_folder, digit_file_name)
                digit.save(digit_path)

                # label 10 is 0 actually
                if label == 10:
                    label = 0

                # Write label to the text file
                labels_file.write(f"{digit_file_name}:{label}\n")


def main():
    # extract images and save to under corresponding file
    download_test_dataset()
    extract_images()


if __name__ == '__main__':
    main()
