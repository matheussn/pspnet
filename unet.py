import argparse
import os

from matplotlib import pyplot
from matplotlib.colors import NoNorm
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.optimizer_v2.adam import Adam

from utils.dataset import load_real_samples
from utils.gpu import set_gpu_limit
import tensorflow as tf


def u_net(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_1")(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_1")(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_3")(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_4")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_2")(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_5")(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_6")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_3")(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_7")(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_8")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="max_pooling_2d_4")(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv_2d_9")(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(filepath=pretrained_weights)

    return model


def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n):
        predict_img = examples[i]
        predict_img[predict_img > 0.5] = 1
        predict_img[predict_img <= 0.5] = 0
        pyplot.imshow(predict_img, cmap='gray', norm=NoNorm())
        filename = f'generated_plot_e%03d.png' % i
        pyplot.savefig(filename)
    pyplot.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    physical_devices = tf.config.experimental.list_physical_devices('CPU')
    print("Num GPUs Available", len(physical_devices))
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset_path', help='path of dataset', default=False)
    parser.add_argument('-tfds', dest='dataset_name', help='Name of tensorflow dataset', default=False)
    args = parser.parse_args()
    set_gpu_limit()

    dataset = load_real_samples(args.dataset_path, target_size=(256, 256))

    model = u_net(input_size=(256, 256, 3))
    img = model(dataset)

    save_plot(img, 10, n=4)
