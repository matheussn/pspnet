from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img

tfds.disable_progress_bar()
auto_tune = tf.data.AUTOTUNE

orig_img_size = (286, 286)
input_img_size = (256, 256, 3)

buffer_size = 286
batch_size = 8


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=[*input_img_size])
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img


def load_dataset(name: str):
    dataset = tfds.load(name, as_supervised=True)

    train_a = dataset['trainA']

    train_images = []
    for samples in train_a:
        img = np.asarray(preprocess_train_image(samples[0], None))
        train_images.append(img)

    train_x = np.asarray(train_images)
    train_x.astype('float32')
    return train_x


def load_real_samples(path: str, target_size: tuple):
    images_glob = glob(f'{path}*')

    train_images = []
    for img_path in images_glob:
        img = np.array(load_img(path=img_path, color_mode='rgb', target_size=target_size))
        train_images.append(img)

    train_x = np.asarray(train_images)
    # convert from unsigned ints to floats
    dataset = train_x.astype('float32')
    # scale from [0,255] to [-1,1]
    dataset = (dataset - 127.5) / 127.5
    return dataset
