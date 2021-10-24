import argparse
import os
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense, Conv2DTranspose, Reshape, Multiply, \
    Embedding, Conv2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model

z_dim = 100
num_class = 4
img_shape = (64, 64, 3)
batch_size = 16
epochs = 10
sample_interval = 1


def build_generator(z_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(256, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, 3, activation='tanh', padding='same'))

    return model


def build_cgan_generator(z_dim, num_classes, base_dir):
    # latent input
    z = Input(shape=(z_dim,))
    # label input
    label = Input(shape=(1,), dtype='int32')
    # convert label to embedding
    label_embedding = Embedding(num_classes, z_dim)(label)

    label_embedding = Flatten()(label_embedding)
    # dot product two inputs
    joined_representation = Multiply()([z, label_embedding])

    generator = build_generator(z_dim)

    conditioned_img = generator(joined_representation)

    model = Model([z, label], conditioned_img)
    # save model blueprint to image
    plot_model(model, f'./{base_dir}/cgan_generator.jpg', show_shapes=True, show_dtype=True)

    return model


def build_discriminator(img_shape, with_drop_out=False):
    model = Sequential(name="discriminator")
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(img_shape[0], img_shape[1], img_shape[2] * 2)))
    model.add(LeakyReLU(alpha=0.2))
    if with_drop_out:
        model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    if with_drop_out:
        model.add(Dropout(0.4))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    if with_drop_out:
        model.add(Dropout(0.4))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    if with_drop_out:
        model.add(Dropout(0.4))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_cgan_discriminator(img_shape, num_classes, base_dir, with_drop_out=False):
    # image input
    img = Input(shape=img_shape)
    # label input
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_classes, np.prod(img_shape), input_length=1)(label)

    label_embedding = Flatten()(label_embedding)

    label_embedding = Reshape(img_shape)(label_embedding)
    # concatenate the image and label
    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = build_discriminator(img_shape, with_drop_out)

    classification = discriminator(concatenated)

    model = Model([img, label], classification)

    plot_model(model, f'./{base_dir}/cgan_discriminator.jpg', show_shapes=True, show_dtype=True)

    return model


def build_cgan(generator, discriminator, z_dim):
    # Random noise vector z
    z = Input(shape=(z_dim,))

    # Image label
    label = Input(shape=(1,))

    # Generated image for that label
    img = generator([z, label])

    classification = discriminator([img, label])

    # Combined Generator -> Discriminator model
    # G([z, lablel]) = x*
    # D(x*) = classification
    model = Model([z, label], classification)

    return model


def bgr_to_rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


def get_label(file: str, labels: dict):
    match = re.search('[a-z]+', file).group(0)
    return labels[match]


def load_all_imgs(path, img_size, dict_label, mode, pre_process=0):
    imgs = []
    labels = []
    files = os.listdir(path)
    files.sort()
    print(f'\n{path} found {len(files)} img to load')
    for index, file in enumerate(files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, mode)

        if mode == 0:
            img = np.expand_dims(img, 2)

        if mode == 1:
            img = bgr_to_rgb(img)
        img = cv2.resize(img, img_size)
        if pre_process == 0:
            img = (img / 127.5) - 1.
        else:
            img = img / 255.
        imgs.append(img)
        labels.append(get_label(file, dict_label))
    return np.array(imgs), np.array(labels)


def sample_images(generator, epoch, base_dir, pos_proccess=0, image_grid_rows=2, image_grid_columns=2):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    labels = np.arange(0, 4).reshape(-1, 1)

    gen_imgs = generator.predict([z, labels])

    fig, axs = plt.subplots(image_grid_rows, image_grid_columns)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            img = gen_imgs[cnt]
            if pos_proccess == 0:
                img = (img + 1) / 2.0
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" % labels[cnt])
            cnt += 1
    filename = f'./{base_dir}/generated_plot_e%03d.jpg' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, base_dir):
    accuracies = []
    losses = []
    log = open(f'{base_dir}/logs.txt', 'a')
    (X_train, y_train) = img, labels

    bat_per_epo = int(X_train.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    real = np.ones((batch_size, 1))  # np.asarray([.9] * batch_size)
    fake = np.zeros((batch_size, 1))  # np.asarray([.1] * batch_size)

    for epoch in range(epochs):
        g_loss = 0
        d_loss = []
        for j in range(bat_per_epo):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator.predict([z, labels])

            # Train the Discriminator
            d_loss_real = discriminator.train_on_batch([imgs, labels], real)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
            #             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Generate a batch of noise vectors
            z = np.random.normal(0, 1, (batch_size, z_dim))

            # Get a batch of random labels
            labels = np.random.randint(0, num_class, batch_size).reshape(-1, 1)

            # Train the Generator
            g_loss = cgan.train_on_batch([z, labels], real)
            print('>%d, %d/%d, [D loss_f: %f, acc_f: %.2f%%, loss_r:%f, acc_r:%.2f%%] [G loss: %f]' % (
                epoch + 1, j + 1, bat_per_epo, d_loss_fake[0], 100 * d_loss_fake[1], d_loss_real[0],
                100 * d_loss_real[1],
                g_loss), file=log)

        if (epoch + 1) % sample_interval == 0:
            # Output training progress

            # Save losses and accuracies so they can be plotted after training
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # Output sample of generated images
            sample_images(generator, epoch, base_dir)

    return accuracies, losses


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels


def train_noise_label_v2(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, base_dir):
    accuracies = []
    losses = []
    log = open(f'{base_dir}/logs.txt', 'a')
    (X_train, y_train) = img, labels

    bat_per_epo = int(X_train.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    real = smooth_labels(np.ones((batch_size, 1)))
    fake = smooth_labels(np.zeros((batch_size, 1)))

    for epoch in range(epochs):
        g_loss = 0
        d_loss = []
        for j in range(bat_per_epo):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator.predict([z, labels])

            # Train the Discriminator
            d_loss_real = discriminator.train_on_batch([imgs, labels], real)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Generate a batch of noise vectors
            z = np.random.normal(0, 1, (batch_size, z_dim))

            # Get a batch of random labels
            labels = np.random.randint(0, num_class, batch_size).reshape(-1, 1)

            # Train the Generator
            g_loss = cgan.train_on_batch([z, labels], real)

        if (epoch + 1) % sample_interval == 0:
            # Output training progress

            print('>%d, %d/%d, [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (
                epoch + 1, j + 1, bat_per_epo, d_loss[0], 100 * d_loss[1], g_loss),
                  file=log)

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # Output sample of generated images
            sample_images(generator, epoch, base_dir)


def run_basic_cgan():
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=4e-4), metrics=['accuracy'])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4))
    train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_with_label_smoothing():
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_with_label_smoothing"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=4e-4), metrics=['accuracy'])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4))
    train_noise_label_v2(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_with_label_smoothing_v2():
    from tensorflow.keras.losses import BinaryCrossentropy
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_with_label_smoothing_v2"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss=BinaryCrossentropy(label_smoothing=.1), optimizer=Adam(learning_rate=4e-4),
                          metrics=['accuracy'])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss=BinaryCrossentropy(label_smoothing=.1), optimizer=Adam(learning_rate=1e-4))
    train_noise_label_v2(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_v2():
    from tensorflow.keras.losses import BinaryCrossentropy
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_v2"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss=BinaryCrossentropy(label_smoothing=.1), optimizer=Adam(learning_rate=4e-4),
                          metrics=['accuracy'])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss=BinaryCrossentropy(label_smoothing=.1), optimizer=Adam(learning_rate=1e-4))
    train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_more_lr():
    from tensorflow.keras.losses import BinaryCrossentropy
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_more_lr"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss=BinaryCrossentropy(label_smoothing=.1), optimizer=Adam(learning_rate=5e-4),
                          metrics=['accuracy'])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss=BinaryCrossentropy(label_smoothing=.1), optimizer=Adam(learning_rate=5e-5))
    train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_more_lr_v2():
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_more_lr_v2"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-4), metrics=['accuracy'])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-5))
    train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_binary_accuracy():
    from tensorflow.keras.metrics import BinaryAccuracy
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_binary_accuracy"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-4), metrics=[BinaryAccuracy()])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-5))
    train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


def run_basic_cgan_binary_accuracy_v2():
    from tensorflow.keras.metrics import BinaryAccuracy
    keras.backend.clear_session()
    BASE_DIR = f"run_basic_cgan_binary_accuracy_v2"

    os.mkdir(BASE_DIR)
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    img, labels = load_all_imgs('./dataset/all_class/', img_shape[:2], classes, 1)
    generator = build_cgan_generator(z_dim, num_class, BASE_DIR)

    discriminator = build_cgan_discriminator(img_shape, num_class, BASE_DIR)
    discriminator.compile(loss='mse', optimizer='sgd',
                          metrics=[BinaryAccuracy()])
    discriminator.trainable = False

    cgan = build_cgan(generator, discriminator, z_dim)
    cgan.compile(loss='mse', optimizer='sgd')
    train(epochs, batch_size, sample_interval, img, labels, generator, discriminator, cgan, BASE_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-type', dest='type', required=True)
    args = parser.parse_args()
    print("ué1", args.type)
    if int(args.type) == 0:
        print("ué")
        run_basic_cgan()
    if int(args.type) == 1:
        print("run_basic_cgan_with_label_smoothing")
        run_basic_cgan_with_label_smoothing()
    if int(args.type) == 2:
        run_basic_cgan_with_label_smoothing_v2()
    if int(args.type) == 3:
        run_basic_cgan_v2()
    if int(args.type) == 4:
        run_basic_cgan_more_lr()
    if int(args.type) == 5:
        run_basic_cgan_more_lr_v2()
    if int(args.type) == 6:
        run_basic_cgan_binary_accuracy()
    if int(args.type) == 7:
        run_basic_cgan_binary_accuracy_v2()
