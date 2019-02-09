
from skimage.io import imread, imsave
import SimpleITK as sitk
import glob
import os
import cv2
import numpy as np

data_path = r'F:\munmun1\Images\data'
gt_path = r'F:\munmun1\Images\gt'

files = glob.glob(os.path.join(data_path, '*.jpg'))
gt_files = glob.glob(os.path.join(gt_path, '*.png'))
image_count = 0

for image, gt in zip(files, gt_files):
    original_image = imread(image)
    gt_image = imread(gt)
    gt_image = gt_image[:, :, 0]
    z = 0
    original_image = np.pad(original_image, ((0, 128), (0, 142), (0, 0)), 'constant', constant_values = 225)
    gt_image = np.pad(gt_image, ((0, 128), (0, 142)), 'constant', constant_values = 24)
    for i in range(0, 769, 64):
        for j in range(0, 1793, 64):
            temp_image = original_image[i: i+256, j: j+256]
            temp_gt = gt_image[i: i+256, j: j+256]
            if temp_gt.shape[0] == 0:
                print('broken image')
                print((i, j), (i+256, j+256))
                input()
            cv2.imwrite(os.path.join(r'F:\munmun1\patches\data', str(image_count) + '_' + str(z) + '.jpg'), temp_image)
            cv2.imwrite(os.path.join(r'F:\munmun1\patches\gt', str(image_count) + '_' + str(z) + '.png'), temp_gt)
            z+=1
    image_count+=1
            