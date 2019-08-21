import random
import pydicom
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from modules.mask_functions import merge_rles, rle2mask, mask2rle
import cv2


def restruct(src, dest):
    """
    Puts all data directly under dest folder

    src: source folder. All data will be searched recursively in it
    dest: destination folder
    """
    for fn in src.glob('**/*dcm'):
        ds = pydicom.dcmread(str(fn))
        pydicom.dcmwrite(str(dest/fn.name), ds)


def create_train(full_size_path, path, size):
    """
    Create size-specific train folder with resized images

    full_size_path: path to train folder with full size images
    path: path to destination folder
    size: size to which images are to be resize
    """
    for fn in full_size_path.glob('**/*dcm'):
        ds = pydicom.dcmread(str(fn))
        img = ds.pixel_array
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        ds.decompress()
        ds.PixelData = img.tobytes()
        ds.Rows, ds.Columns = img.shape
        pydicom.dcmwrite(str(path/fn.name), ds)


def change_csv(old, new, path, size=256):
    """
    Change mask csv to match resizd images

    old: path to base csv for full size images
    new: path to new csv for resized images
    path: path to train folder
    size: size to which masks are to be resized
    """
    df = pd.read_csv(old, sep=', ')
    new_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        image_id = row.ImageId
        label = row.EncodedPixels
        image_id = Path(path.name)/(image_id+'.dcm')
        mask = rle2mask(label, 1024, 1024)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
        mask = (mask > 127).astype(np.uint8)*255
        label = mask2rle(mask.T, size, size)
        new_df.loc[row.Index] = [image_id, label]
    new_df.to_csv(new, index=False)


def merge_doubles(old, new):
    """
    Merge rles in old csv that fit in same image

    old: path to csv to merge rles in
    new: path to save new merge csv
    """
    df = pd.read_csv(old)
    new_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for k, id in enumerate(df['ImageId'].unique()):
        new_rle = ''
        for rle in df.loc[df['ImageId'] == id, 'EncodedPixels']:
            new_rle = merge_rles(new_rle, rle)
        new_df.loc[k] = [id, new_rle]
    new_df.to_csv(new, index=False)


def keep_pos(old, new, pct=0.2):
    """
    Create a new csv with only a random portion of empty masks

    old: path to base csv
    new: path to save new csv
    pct: probability to keep an empty mask
    """
    df = pd.read_csv(old)
    new_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for row in df.itertuples():
        id = row.ImageId
        rle = row.EncodedPixels
        k = row.Index
        if rle != '-1' or random.random() <= pct:
            new_df.loc[k] = [id, rle]
    new_df.to_csv(new, index=False)


def create_classif_csv(old, new):
    """
    Create a csv suited for classification (with 0 and 1 labels)

    old: path to segmentation csv
    new: path to save new classification csv
    """
    df = pd.read_csv(old)
    new_df = pd.DataFrame(columns=['ImageId', 'Labels'])
    for row in df.itertuples():
        image_id = row.ImageId
        rle = row.EncodedPixels
        new_df.loc[row.Index] = [image_id, 1 if rle != '-1' else 0]
    new_df.to_csv(new, index=False)


def open_image(fn):
    """
    Opens a dicom image as np array

    fn: path to dicom file
    """
    return pydicom.dcmread(str(fn)).pixel_array


def show(img, figsize=(10, 10)):
    """
    Show image with color map bone

    img: image as np array
    figsize: size of the figure
    """
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.bone)
    plt.show()


def show_dcm_info(fn):
    """
    Display dicom metadata of a file

    fn: path to dicom file
    """
    dataset = pydicom.dcmread(str(fn))
    print("Filename.........:", fn)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


def getNextFilePath(output_folder, base_name):
    """
    Gets highest indice of all files with name base_name in output_folder

    output_folder: folder in which to look for files
    base_name: base of file names

    return: next available index
    """
    highest_num = 0
    for f in output_folder.iterdir():
        if f.is_file():
            try:
                f = str(f.with_suffix('').name)
                if f.split('_')[:-1] == base_name.split('_'):
                    split = f.split('_')
                    file_num = int(split[-1])
                    if file_num > highest_num:
                        highest_num = file_num
            except ValueError:
                'The file name "%s" is incorrect. Skipping' % f

    output_file = highest_num + 1
    return output_file
