import os
from pathlib import Path

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

from skimage.io import imread


def PlateDay1():
    return imread(current_file_dir / 'StandardDay1.jpg')


def PlateDay6():
    return imread(current_file_dir / 'StandardDay6.jpg')


def PlateSeries():
    series = []
    fnames = os.listdir(current_file_dir / 'PlateSeries')
    fnames.sort()
    for fname in fnames:
        series.append(imread(current_file_dir / 'PlateSeries' / fname))
    return series
