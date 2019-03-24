from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as B

smooth = 1


def dice(y_true, y_pred):
    y_true_f = B.flatten(y_true)
    y_pred_f = B.flatten(y_pred)
    intersect = B.sum(y_true_f * y_pred_f)
    return (2. * intersect + smooth) / (B.sum(y_pred_f) + B.sum(y_true_f) + smooth)


def loss_dice(y_true, y_pred):
    return -dice(y_true, y_pred)


def unet_architecture(IMG_W=256, IMG_H=256, IMG_CHANNELS=3):
    inputs = Input((IMG_H, IMG_W, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)
    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    dp1 = Dropout(0.1)(conv1)
    conv2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dp1)
    pool1 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
    dp2 = Dropout(0.1)(conv3)
    conv4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dp2)
    pool2 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
    dp3 = Dropout(0.2)(conv5)
    conv6 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(dp3)
    pool3 = MaxPooling2D((2, 2)) (conv6)
