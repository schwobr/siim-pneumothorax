import numpy as np


def mask2rle(img, width, height):
    """
    Used to convert mask into run-length encoded string (column-first)
    with relative values.

    img: transposed version of the mask to encode
    width: width of the mask
    height: height of the mask

    return: rle string corresponding to the mask
    """
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle) if rle != [] else '-1'


def rle2mask(rle, width, height):
    """
    Converts run-length encoded string with relative values
    (column-first) to corresponding mask

    rle: run-length encoded string
    width: width of the mask image
    height: height of the mask image

    return: corresponding mask
    """
    if rle == '-1':
        return np.zeros((width, height))
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T


def absol2relat(rle):
    """
    Converts a run-length encoded string with absolute values
    to one with relative values

    rle: rle string with absolute values

    return: rle string with relative values
    """
    if rle == '-1':
        return '-1'
    pixels = rle.split()
    new_rle = []
    cur = 0
    for k in range(0, len(pixels), 2):
        if k == 0:
            new_rle.append(pixels[k])
            new_rle.append(pixels[k+1])
        else:
            cur = int(pixels[k])
            prev = int(pixels[k-2])+int(pixels[k-1])
            new_rle.append(str(cur-prev))
            new_rle.append(pixels[k+1])
    return ' '.join(new_rle)


def relat2absol(rle):
    """
    Converts a run-length encoded string with relative values
    to one with absolute values

    rle: rle string with relative values

    return: rle string with absolute values
    """
    if rle == '-1':
        return '-1'
    pixels = rle.split()
    new_rle = []
    cur = 0
    for k in range(0, len(pixels), 2):
        pix = pixels[k]
        cur += int(pix)
        length = pixels[k+1]
        new_rle.append(str(cur))
        new_rle.append(length)
        cur += int(length)
    return ' '.join(new_rle)


def merge_rles(rle1, rle2):
    """
    Merges 2 absolute run-length encoded strings

    rle1: first rle string
    rle2: second rle string

    return: New rle corresponding to the merge of the inputs
    """
    if rle1 == rle2:
        return rle1
    i1 = 0
    i2 = 0
    rle = []
    pixels1 = relat2absol(rle1).split()
    pixels2 = relat2absol(rle2).split()
    while i1 < len(pixels1) and i2 < len(pixels2):
        p1 = int(pixels1[i1])
        l1 = int(pixels1[i1+1])
        p2 = int(pixels2[i2])
        l2 = int(pixels2[i2+1])
        if p1 <= p2:
            rle.append(str(p1))
            if p2 <= p1+l1-1:
                rle.append(str(max(p2-p1+l2, l1)))
                i2 += 2
            else:
                rle.append(str(l1))
            i1 += 2
        else:
            rle.append(str(p2))
            if p1 <= p2+l2-1:
                rle.append(str(max(p1-p2+l1, l2)))
                i1 += 2
            else:
                rle.append(str(l2))
            i2 += 2

    rle += pixels1[i1:]+pixels2[i2:]
    return absol2relat(' '.join(rle))
