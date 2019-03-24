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

# seed = 42
# random.seed = seed
# np.random.seed = seed

TRAIN_PATH = 'D:/Projects/starik/stage1_train'
TEST_PATH = 'D:/Projects/starik/stage1_test'

tr_ids = next(os.walk(TRAIN_PATH))[1]
tst_ids = next(os.walk(TEST_PATH))[1]


def retrieve_train(IMG_W=256, IMG_H=256, IMG_CHANNELS=3):
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
        np.save("train_img", X_train)
        np.save("train_mask", Y_train)
        return X_train, Y_train


def retrieve_test(IMG_W=256, IMG_H=256, IMG_CHANNELS=3):
    X_test = np.zeros((len(tst_ids), IMG_H, IMG_W, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()
    if os.path.isfile("test_img.npy") and os.path.isfile("test_size.npy"):
        print("Test file loaded from memory")
        X_test = np.load("test_img.npy")
        sizes_test = np.load("test_size.npy")
        return X_test, sizes_test
    b = Progbar(len(tst_ids))
    for n, id_ in enumerate(tst_ids):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_H, IMG_W), mode='constant', preserve_range=True)
        X_test[n] = img
        b.update(n)
    np.save("test_img", X_test)
    np.save("test_size", sizes_test)
    return X_test, sizes_test


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def mask_to_rle(preds_test_upsampled):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(tst_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    return new_test_ids, rles


if __name__ == '__main__':
    x, y = retrieve_train()
    x, y = retrieve_test()
