# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:58:16 2019

@author: cropdata
"""

import cv2
image = r"C:\Users\COE1\ISBI_CHAOS\Images\data\1.jpg"
gt = r"C:\Users\COE1\ISBI_CHAOS\Images\data"

original_image = imread(image)
#gt_image = imread(gt)
#gt_image = gt_image[:, :, 0]
#z = 0
soln = np.zeros([1024, 2048])
original_image = np.pad(original_image, ((0, 128), (0, 142), (0, 0)), 'constant', constant_values = 225)
#gt_image = np.pad(gt_image, ((0, 128), (0, 142)), 'constant', constant_values = 24)
for i in range(0, 769, 128):
    for j in range(0, 1793, 128):
        temp_image = original_image[i: i+256, j: j+256]
        #temp_image /= 255
        temp_image = temp_image.squeeze() / 255
        soln[i: i+256, j: j+256] = model.predict(temp_image.reshape((1, 256, 256, 3))).reshape((256, 256))
        #soln = model.predict(temp_image.reshape((1, 256, 256, 3))).reshape((256, 256))
soln = soln[:896, :1906]
imshow(soln)
cv2.imwrite(gt, soln+'soln.png')