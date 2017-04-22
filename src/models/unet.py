# Unet implementation based on https://github.com/jocicmarko/ultrasound-nerve-segmentation
import numpy as np
np.random.seed(865)

from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Conv2DTranspose, Lambda, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from scipy.misc import imsave
from os import path, makedirs
import argparse
import keras.backend as K
import logging
import pickle
import tifffile as tiff

import sys
sys.path.append('.')
from src.utils.runtime import funcname, gpu_selection
from src.utils.model import dice_coef, dice_coef_loss, KerasHistoryPlotCallback, KerasSimpleLoggerCallback, \
    jaccard_coef, jaccard_coef_int
from src.utils.data import random_transforms
from src.utils.isbi_utils import isbi_get_data_montage


class UNet():

    def __init__(self, checkpoint_name):

        self.config = {
            'data_path': 'data',
            'input_shape': (64, 64),
            'output_shape': (64, 64),
            'transform_train': True,
            'batch_size': 64,
            'nb_epoch': 120
        }

        self.checkpoint_name = checkpoint_name
        self.net = None
        self.imgs_trn = None
        self.msks_trn = None
        self.imgs_val = None
        self.msks_val = None

        return

    @property
    def checkpoint_path(self):
        return 'checkpoints/%s_%d' % (self.checkpoint_name, self.config['input_shape'][0])

    def load_data(self):

        self.imgs_trn, self.msks_trn = isbi_get_data_montage('data/train-volume.tif', 'data/train-labels.tif',
                                                             nb_rows=6, nb_cols=5, rng=np.random)
        self.imgs_val, self.msks_val = isbi_get_data_montage('data/train-volume.tif', 'data/train-labels.tif',
                                                             nb_rows=5, nb_cols=6, rng=np.random)

        imsave('%s/trn_imgs.png' % self.checkpoint_path, self.imgs_trn)
        imsave('%s/trn_msks.png' % self.checkpoint_path, self.msks_trn)
        imsave('%s/val_imgs.png' % self.checkpoint_path, self.imgs_val)
        imsave('%s/val_msks.png' % self.checkpoint_path, self.msks_val)
        return

    def compile(self):

        K.set_image_dim_ordering('tf')

        x = inputs = Input(shape=self.config['input_shape'], dtype='float32')

        x = Reshape(self.config['input_shape'] + (1,))(x)
        x = Conv2D(32,  3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(32,  3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = dc_0_out = Dropout(0.2)(x)

        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = dc_1_out = Dropout(0.2)(x)

        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = dc_2_out = Dropout(0.2)(x)

        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = dc_3_out = Dropout(0.2)(x)

        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2DTranspose(256, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
        x = concatenate([x, dc_3_out])
        x = Dropout(0.2)(x)

        x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2DTranspose(128, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
        x = concatenate([x, dc_2_out])
        x = Dropout(0.2)(x)

        x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2DTranspose(64, 2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
        x = concatenate([x, dc_1_out])
        x = Dropout(0.2)(x)

        x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2DTranspose(32,  2, strides=2, activation='relu', kernel_initializer='he_normal')(x)
        x = concatenate([x, dc_0_out])
        x = Dropout(0.2)(x)

        x = Conv2D(32,  3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(32,  3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(2, 1, activation='softmax')(x)
        x = Lambda(lambda x: x[:, :, :, 1], output_shape=self.config['output_shape'])(x)

        self.net = Model(inputs=inputs, outputs=x)
        self.net.compile(optimizer=Adam(lr=0.0005), loss='binary_crossentropy', metrics=[dice_coef])

        return

    def train(self):

        logger = logging.getLogger(funcname())

        gen_trn = self.batch_gen_trn(imgs=self.imgs_trn, msks=self.msks_trn, batch_size=self.config[
            'batch_size'], transform=self.config['transform_train'])
        gen_val = self.batch_gen_trn(imgs=self.imgs_val, msks=self.msks_val, batch_size=self.config[
            'batch_size'], transform=self.config['transform_train'])

        cb = [
            ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, cooldown=3, min_lr=1e-5, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, cooldown=3, min_lr=1e-5, verbose=1),
            EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=1, mode='min'),
            ModelCheckpoint(self.checkpoint_path + '/weights_loss_val.weights',
                            monitor='val_loss', save_best_only=True, verbose=1),
            ModelCheckpoint(self.checkpoint_path + '/weights_loss_trn.weights',
                            monitor='loss', save_best_only=True, verbose=1)
        ]

        logger.info('Training for %d epochs.' % self.config['nb_epoch'])

        self.net.fit_generator(generator=gen_trn, steps_per_epoch=100, epochs=self.config['nb_epoch'],
                               validation_data=gen_val, validation_steps=20, verbose=1, callbacks=cb)

        return

    def batch_gen_trn(self, imgs, msks, batch_size, transform=False, rng=np.random):

        H, W = imgs.shape
        wdw_H, wdw_W = self.config['input_shape']
        _mean, _std = np.mean(imgs), np.std(imgs)
        normalize = lambda x: (x - _mean) / (_std + 1e-10)

        while True:

            img_batch = np.zeros((batch_size,) + self.config['input_shape'], dtype=imgs.dtype)
            msk_batch = np.zeros((batch_size,) + self.config['output_shape'], dtype=msks.dtype)

            for batch_idx in range(batch_size):

                # Sample a random window.
                y0, x0 = rng.randint(0, H - wdw_H), rng.randint(0, W - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W

                img_batch[batch_idx] = imgs[y0:y1, x0:x1]
                msk_batch[batch_idx] = msks[y0:y1, x0:x1]

                if transform:
                    [img_batch[batch_idx], msk_batch[batch_idx]] = random_transforms(
                        [img_batch[batch_idx], msk_batch[batch_idx]])

            img_batch = normalize(img_batch)
            yield img_batch, msk_batch

    def predict(self, imgs):
        imgs = (imgs - np.mean(imgs)) / (np.std(imgs) + 1e-10)
        return self.net.predict(imgs).round()


def main():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(funcname())

    prs = argparse.ArgumentParser()
    prs.add_argument('--name', help='name used for checkpoints', default='unet', type=str)

    subprs = prs.add_subparsers(title='actions', description='Choose from one of the actions.')
    subprs_trn = subprs.add_parser('train', help='Run training.')
    subprs_trn.set_defaults(which='train')
    subprs_trn.add_argument('-w', '--weights', help='path to keras weights')

    subprs_sbt = subprs.add_parser('submit', help='Make submission.')
    subprs_sbt.set_defaults(which='submit')
    subprs_sbt.add_argument('-w', '--weights', help='path to keras weights', required=True)
    subprs_sbt.add_argument('-t', '--tiff', help='path to tiffs', default='data/test-volume.tif')

    args = vars(prs.parse_args())
    assert args['which'] in ['train', 'submit']

    model = UNet(args['name'])

    if not path.exists(model.checkpoint_path):
        makedirs(model.checkpoint_path)

    def load_weights():
        if args['weights'] is not None:
            logger.info('Loading weights from %s.' % args['weights'])
            model.net.load_weights(args['weights'])

    if args['which'] == 'train':
        model.compile()
        load_weights()
        model.net.summary()
        model.load_data()
        model.train()

    elif args['which'] == 'submit':
        out_path = '%s/test-volume-masks.tif' % model.checkpoint_path
        model.config['input_shape'] = (512, 512)
        model.config['output_shape'] = (512, 512)
        model.compile()
        load_weights()
        model.net.summary()
        imgs_sbt = tiff.imread(args['tiff'])
        msks_sbt = model.predict(imgs_sbt)
        logger.info('Writing predicted masks to %s' % out_path)
        tiff.imsave(out_path, msks_sbt)


if __name__ == "__main__":
    main()
