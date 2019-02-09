import numpy as np
import os
import glob

folders = glob.glob(r'C:\Users\COE1\ISBI_CHAOS_challenge\data\CT_data_batch1\CT_data_batch1\*')
comb_folder = r'C:\Users\COE1\ISBI_CHAOS_challenge\comb_data'
i = 0
j = 0
for folder in folders:
    image_files = glob.glob(os.path.join(folder, 'DICOM_anon\*'))
    gt_files = glob.glob(os.path.join(folder, 'Ground\*'))
    new_image_files = os.path.join(comb_folder, 'data')
    gt_image_files = os.path.join(comb_folder, 'gt')
    for file, gt in zip(image_files, gt_files):
        os.rename(file, os.path.join(new_image_files, str(i) + '.dcm'))
        os.rename(gt, os.path.join(gt_image_files, str(i) + '.png'))
        i+=1