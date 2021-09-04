import argparse
import os
import uuid

from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session

from utils.dataset import load_dataset, load_real_samples
from utils.train import train

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(Session(config=config))

BASE_DIR = f"exec_{uuid.uuid1()}"

os.mkdir(BASE_DIR)


# define the standalone discriminator model
def define_discriminator(in_shape=(64, 64, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=1e-3, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim=100):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(256, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, 3, activation='tanh', padding='same'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    opt = Adam(learning_rate=3e-4, beta_1=0.05)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', dest='dataset_path', help='path of dataset', default=False)
    parser.add_argument('-tfds', dest='dataset_name', help='Name of tensorflow dataset', default=False)
    args = parser.parse_args()

    # size of the latent space
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator()
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # load image data
    if type(args.dataset_path) is not bool:
        dataset = load_real_samples(args.dataset_path, target_size=(64, 64))
    else:
        dataset = load_dataset(args.dataset_name)
    # train model
    train(g_model, d_model, gan_model, dataset, BASE_DIR)
