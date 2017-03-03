# Unet implementation based on https://github.com/jocicmarko/ultrasound-nerve-segmentation
from keras.models import Model, load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from skimage.transform import resize
import keras.backend as K
import logging
import numpy as np
import os
import random
import pickle
import tifffile as tiff

import sys;
sys.path.append('.')
from src.utils.runtime import funcname, gpu_selection
from src.utils.model import dice_coef, dice_coef_loss, KerasHistoryPlotCallback, KerasSimpleLoggerCallback
from src.utils.data import random_transforms

class UNetBinary():
    def __init__(self):

        self.config = {
            'checkpoint_path_net': None,
            'checkpoint_path_model': None,
            'checkpoint_path_history': None,
            'data_path': 'data',
            'img_shape': (512, 512, 1),
            'input_shape': (128, 128, 1),       # Row dimension has to be a power of 2.
            'output_shape': (128, 128, 1),
            'transform_train': True,
            'prop_trn': 1.0,
            'prop_val': 0.0,
            'batch_size': 5,
            'nb_epoch': 25,
            'seed': 865
        }

        self.net = None
        self.imgs = None
        self.msks = None
        self.imgs_mean = None
        self.imgs_std = None
        self.history = None

        return

    def load_data(self):

        logger = logging.getLogger(funcname())

        logger.info('Reading images from %s.' % self.config['data_path'])
        self.imgs = tiff.imread('%s/train-volume.tif' % self.config['data_path'])
        self.msks = tiff.imread('%s/train-labels.tif' % self.config['data_path'])

        logger.info('Images: %s, labeld: %s' % (str(self.imgs.shape), str(self.msks.shape)))

        self.imgs_mean = np.mean(self.imgs)
        self.imgs_std = np.std(self.imgs)

        logger.info('Images mean: %.2lf, std: %.2lf' % (self.imgs_mean, self.imgs_std))

        return

    def batch_gen(self, imgs, msks, batch_size, shuffle=False, infinite=False, transform=False):

        logger = logging.getLogger(funcname())

        if msks is None:
            msks = np.zeros(imgs.shape)

        X_batch = np.empty((batch_size,) + self.config['input_shape'])
        Y_batch = np.empty((batch_size,) + self.config['output_shape'])
        batch_idx = 0

        combined = [(img, msk) for img, msk in zip(imgs, msks)]

        while True:

            for img, msk in combined:
                _img = (img - self.imgs_mean) / self.imgs_std
                _img = resize(_img, output_shape=self.config['input_shape'][:2])
                _msk = resize(msk, output_shape=self.config['output_shape'][:2])

                if transform:
                    [_img, _msk] = random_transforms([_img, _msk])

                X_batch[batch_idx] = _img.reshape(self.config['input_shape'])
                Y_batch[batch_idx] = _msk.reshape(self.config['output_shape'])
                batch_idx += 1
                if batch_idx == batch_size:
                    batch_idx = 0
                    yield (X_batch, Y_batch)

            if not infinite:
                break

            # Shuffling useful in training.
            if shuffle:
                random.shuffle(combined)

    def compile(self):

        K.set_image_dim_ordering('tf')

        inputs = Input(shape=self.config['input_shape'])

        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        self.net = Model(input=inputs, output=conv10)

        self.net.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])

        return

    def train(self, notebook=False):

        logger = logging.getLogger(funcname())

        gen_trn = self.batch_gen(imgs=self.imgs, msks=self.msks, batch_size=self.config['batch_size'], \
                                 shuffle=True, infinite=True, transform=self.config['transform_train'])

        callbacks = []

        # callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=2, min_lr=1e-6, verbose=1))

        if self.config['checkpoint_path_net']:
            callbacks.append(ModelCheckpoint(self.config['checkpoint_path_net'], monitor='val_loss', save_best_only=True))

        if notebook:
            callbacks.append(KerasSimpleLoggerCallback())
            callbacks.append(KerasHistoryPlotCallback())

        logger.info('Training for %d epochs with %d train.' % (self.config['nb_epoch'], len(self.imgs)))

        random.seed(self.config['seed'])

        result = self.net.fit_generator(
            nb_epoch=self.config['nb_epoch'],
            samples_per_epoch=len(self.imgs),
            generator=gen_trn,
            initial_epoch=0,
            callbacks=callbacks,
            class_weight='auto',
            verbose=int(notebook==False)
        )

        self.history = result.history

        if self.config['checkpoint_path_history']:
            logger.info('Saving history to %s.' % self.config['checkpoint_path_history'])
            f = open(self.config['checkpoint_path_history'], 'wb')
            pickle.dump(self.history, f)
            f.close()

        return

    def save(self):
        logger = logging.getLogger(funcname())

        if self.config['checkpoint_path_model']:
            logger.info('Saving model to %s.' % self.config['checkpoint_path_model'])
            payload = (self.config, self.mean, self.std)
            f = open(self.config['checkpoint_path_model'], 'wb')
            pickle.dump(payload, f)
            f.close()

        return

    def load(self, checkpoint_path):
        f = open(checkpoint_path, 'rb')
        (config, mean, std) = pickle.load(f)
        f.close()
        self.config = config
        self.mean = mean
        self.std = std
        return

def main():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(funcname())
    gpu_selection()

    model = UNetBinary()
    model.config['checkpoint_path_net'] = 'checkpoints/unet_binary.net'
    model.config['checkpoint_path_model'] = 'checkpoints/unet_binary.model'
    model.config['checkpoint_path_history'] = 'checkpoints/unet_binary.history'

    model.compile()
    model.net.summary()
    model.load_data()
    model.train()
    # model.save()

if __name__ == "__main__":
    main()