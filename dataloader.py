import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pydicom
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

#%%
path_train = r'C:\Users\COE1\ISBI_CHAOS\patches_256x256' 
im_height = 256
im_width = 256
def get_data(path, train=True):
    ids = glob.glob(os.path.join(path, "images", '*.jpg'))
    ids = [os.path.basename(i)[:-4] for i in ids] 
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        x_img = imread(os.path.join(path, 'images', id_ + '.jpg'))
        #x_img = img.pixel_array
        x_img = resize(x_img, (256, 256, 3), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(os.path.join(path, 'gt', id_ + '.png'), grayscale=True))
            mask = resize(mask, (256, 256, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., :] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
    
X, y = get_data(path_train, train=True)
#%%

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

def conv_block(model, dim, activation = 'relu', batch_norm = 'True', residual = 'True', drate = 0.5):
	n = Conv2D(dim, 3, activation = activation, padding='same', kernel_initializer = 'he_normal')(model)
	n = BatchNormalization()(n) if batch_norm else n
	n = Dropout(drate)(n) if drate else n
	n = Conv2D(dim, 3, activation = activation, padding='same', kernel_initializer = 'he_normal')(n)
	n = BatchNormalization()(n) if batch_norm else n
	return Concatenate()([model, n]) if residual else n

def level_block(model, dim, depth, inc, acti, drate, batch_norm, maxpool, upconv, residual):
	if depth > 0:
		n = conv_block(model, dim, acti, batch_norm, residual)
		model = MaxPooling2D()(n) if maxpool else Conv2D(dim, 3, strides=2, padding='same')(n)
		model = level_block(model, int(inc*dim), depth-1, inc, acti, drate, batch_norm, maxpool, upconv, residual)
		if upconv:
			model = UpSampling2D()(model)
			model = Conv2D(dim, 2, activation=acti, padding='same', kernel_initializer = 'he_normal')(model)
		else:
			model = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same', kernel_initializer = 'he_normal')(model)
		n = Concatenate()([n, model])
		model = conv_block(n, dim, acti, batch_norm, residual)
	else:
		model = conv_block(model, dim, acti, batch_norm, residual, drate)
	return model

def UNet(img_shape, out_ch = 1, start_ch = 64, depth = 4, inc_rate = 2, activation = 'relu', 
		 dropout = 0.5, batchnorm = True, maxpool = True, upconv = True, residual = True):
	i = Input(shape = img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

model = UNet((256, 256, 3), out_ch = 1, start_ch = 64, depth = 5, inc_rate = 2, activation = 'relu',
             dropout = 0.5, batchnorm = True, maxpool = True, upconv = True, residual = True)

#%%
import gc
# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2019)
del(X)
del(y)
gc.collect()

#%%
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    


#%%

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy', dice_coef_loss])
model.summary()


#%%
from keras.callbacks import TensorBoard

callbacks = [
    #EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1),
    ModelCheckpoint('./ravi_model-tgs-salt-dropout0.5-diceloss.h5', verbose=1, save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
]
#%%
#model.load_weights(r'C:\Users\COE1\ISBI_CHAOS_challenge\model-tgs-salt-dropout0.5-diceloss.h5')

#%%
results = model.fit(X_train, y_train, batch_size=2, epochs=100, callbacks=callbacks, shuffle = True,
                   validation_data=(X_valid, y_valid))
#%%
temp = model.predict(X_valid[30].reshape((1, 256, 256, 3)))
imshow(temp.reshape((256, 256, 1)))
imshow(X_valid[30].reshape((256, 256, 3)))
imshow(y_valid[30].reshape((256, 256, 1)))
from skimage.io import imsave
imsave('./newer_results.png', temp.reshape((256, 256, 1)))
imsave('./newer_gt.png', y_train[128].reshape((256, 256, 1)))
imsave('./newer_image.png', X_train[128].reshape((256, 256, 3)))
