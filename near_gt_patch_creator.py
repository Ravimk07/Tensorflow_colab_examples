import pydicom
from skimage.io import imread, imsave, imshow
import SimpleITK as sitk
import glob
import os
import cv2
from skimage.measure import regionprops
import numpy as np

data_path = r'C:\Users\COE1\ISBI_CHAOS_challenge\comb_data\data'
gt_path = r'C:\Users\COE1\ISBI_CHAOS_challenge\comb_data\gt'

files = glob.glob(os.path.join(data_path, '*.dcm'))
gt_files = glob.glob(os.path.join(gt_path, '*.png'))


gt = imread(gt_files[468])
n_white_pix = np.sum(gt == 255)
if n_white_pix > 0:
    props = regionprops(gt)
    print(props[0].centroid)
