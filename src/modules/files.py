import pydicom
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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
        rles = []
        for rle in df.loc[df['ImageId'] == id, 'EncodedPixels']:
            rles.append(rle)
        new_df.loc[k] = [id, ' '.join(rles)]
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
