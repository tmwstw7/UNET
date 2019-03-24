import os
import random
import sys
import warnings as w
import numpy as np
from keras.utils import Progbar
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from itertools import chain

w.filterwarnings('ignore', category=UserWarning, module='skimage')

#seed = 42
#random.seed = seed
#np.random.seed = seed

TRAIN_PATH = 'D:/Projects/starik/stage1_train'
TEST_PATH = 'D:/Projects/starik/stage1_test'

tr_ids = next(os.walk(TRAIN_PATH))[1]
tst_ids = next(os.walk(TEST_PATH))[1]


def retrive_train(IMG_W=256, IMG_H=256, IMG_CHANNELS=3):
    X_train = np.zeros((len(tr_ids), IMG_H, IMG_W, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(tr_ids), IMG_H, IMG_W, 1), dtype=np.bool)
    print('Retrieving and resizing train images')
    sys.stdout.flush()
    if os.path.isfile("train_imf.npy") and os.path.isfile("train_mask.npy"):
        print("Train file loaded from memory")
        X_train = np.load("train_img.npy")
        Y_train = np.load("train_mask.npy")
        return X_train, Y_train
    a = Progbar(len(tr_ids))
    for n, id_ in enumerate(tr_ids):
        path = TRAIN_PATH = id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_H, IMG_W), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_H, IMG_W, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_H, IMG_W), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
        a.update(n)

