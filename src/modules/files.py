import random
import pydicom
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from modules.mask_functions import merge_rles


def restruct(src, dest):
    for fn in src.glob('**/*dcm'):
        ds = pydicom.dcmread(str(fn))
        pydicom.dcmwrite(str(dest/fn.name), ds)


def change_csv(old, new, path):
    df = pd.read_csv(old, sep=', ')
    new_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for row in df.itertuples():
        image_id = row.ImageId
        label = row.EncodedPixels
        image_id = Path(path.name)/(image_id+'.dcm')
        new_df.loc[row.Index] = [image_id, label]
    new_df.to_csv(new, index=False)


def merge_doubles(old, new):
    df = pd.read_csv(old)
    new_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for k, id in enumerate(df['ImageId'].unique()):
        new_rle = ''
        for rle in df.loc[df['ImageId'] == id, 'EncodedPixels']:
            new_rle = merge_rles(new_rle, rle)
        new_df.loc[k] = [id, new_rle]
    new_df.to_csv(new, index=False)


def keep_pos(old, new):
    df = pd.read_csv(old)
    new_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    for row in df.itertuples():
        id = row.ImageId
        rle = row.EncodedPixels
        k = row.Index
        if rle != '-1' or random.random() <= 0.2:
            # keep all postive and only 20% of negative for segmentation
            new_df.loc[k] = [id, rle]
    new_df.to_csv(new, index=False)


def create_classif_csv(old, new):
    df = pd.read_csv(old)
    new_df = pd.DataFrame(columns=['ImageId', 'Labels'])
    for row in df.itertuples():
        image_id = row.ImageId
        rle = row.EncodedPixels
        new_df.loc[row.Index] = [image_id, 1 if rle != '-1' else 0]
    new_df.to_csv(new, index=False)


def open_image(fn):
    return pydicom.dcmread(str(fn)).pixel_array


def show(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.bone)
    plt.show()


def show_dcm_info(fn):
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
