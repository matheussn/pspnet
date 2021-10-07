import numpy as np
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense, Conv2DTranspose, Reshape, Multiply, \
    Embedding, Conv2D, Concatenate
from tensorflow.python.keras.utils.vis_utils import plot_model


def build_generator(z_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(256, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(256, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, 3, activation='tanh', padding='same'))

    return model


def build_cgan_generator(z_dim, num_classes):
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
    plot_model(model, './exec_cgan/cgan_generator.jpg', show_shapes=True, show_dtype=True)

    return model


def build_discriminator(img_shape):
    model = Sequential(name="discriminator")
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(img_shape[0], img_shape[1], img_shape[2] * 2)))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
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
    # model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model


def build_cgan_discriminator(img_shape, num_classes):
    # image input
    img = Input(shape=img_shape)
    # label input
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(num_classes, np.prod(img_shape), input_length=1)(label)

    label_embedding = Flatten()(label_embedding)

    label_embedding = Reshape(img_shape)(label_embedding)
    # concatenate the image and label
    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = build_discriminator(img_shape)

    classification = discriminator(concatenated)

    model = Model([img, label], classification)

    plot_model(model, './exec_cgan/cgan_discriminator.jpg', show_shapes=True, show_dtype=True)

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
