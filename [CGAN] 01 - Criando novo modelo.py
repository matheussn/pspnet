import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (
    Concatenate, Dense, Multiply,
    Embedding, Flatten, Input, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Activation)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from dataset_utils import load_all_imgs

# Define important parameters
img_shape = (64, 64, 3)
z_dim = 100
n_class = 10


# Generator CNN model
def generator_model(z_dim):
    model = Sequential()

    model.add(Dense(256 * 16 * 16, input_dim=z_dim, ))
    model.add(Reshape((16, 16, 256)))

    model.add(Conv2DTranspose(128, 3, 2, padding='same', ))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64, 3, 1, padding='same', ))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(3, 3, 2, padding='same', ))
    model.add(Activation('tanh'))

    return model


# generator input
def generator(z_dim):
    # latent input
    z = Input(shape=(z_dim,))
    # label input
    label = Input(shape=(1,), dtype='int32')
    # convert label to embedding
    label_embedding = Embedding(n_class, z_dim)(label)

    label_embedding = Flatten()(label_embedding)
    # dot product two inputs
    joined_representation = Multiply()([z, label_embedding])

    generator = generator_model(z_dim)

    conditioned_img = generator(joined_representation)

    model = Model([z, label], conditioned_img)
    # save model blueprint to image
    plot_model(model, './exec_cgan/cgan_generator.jpg', show_shapes=True, show_dtype=True)

    return model


# discriminator CNN model
def discriminator_model(img_shape):
    model = Sequential()

    model.add(Conv2D(64, 3, 2, input_shape=(img_shape[0], img_shape[1], img_shape[2] * 2), ))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(64, 3, 2, input_shape=img_shape, padding='same', ))
    model.add(LeakyReLU(alpha=0.001))

    model.add(Conv2D(128, 3, 2, input_shape=img_shape, padding='same', ))
    model.add(LeakyReLU(alpha=0.001))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


def discriminator(img_shape):
    # image input
    img = Input(shape=img_shape)
    # label input
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(n_class, np.prod(img_shape), input_length=1)(label)

    label_embedding = Flatten()(label_embedding)

    label_embedding = Reshape(img_shape)(label_embedding)
    # concatenate the image and label
    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = discriminator_model(img_shape)

    classification = discriminator(concatenated)

    model = Model([img, label], classification)

    plot_model(model, './exec_cgan/cgan_discriminator.jpg', show_shapes=True, show_dtype=True)

    return model


# define a complete GAN architecture
def cgan(generator, discriminator):
    z = Input(shape=(z_dim,))

    label = Input(shape=(1,))

    img = generator([z, label])

    classification = discriminator([img, label])

    model = Model([z, label], classification)

    return model


discriminator = discriminator(img_shape)
# compile the discriminator architecture 
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

generator = generator(z_dim)
# set discriminator to non-trainanle 
discriminator.trainable = False
# compile the whole C-GAN architectu
cgan = cgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer=Adam())

# label to category dictionary
classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
dict_clothes = {0: "healthy", 1: "mild", 2: "moderate", 3: "severe"}


# function to plot and save sample images
def plot_sample_images(epoch, rows=2, columns=2):
    z = np.random.normal(0, 1, (rows * columns, z_dim))
    a = np.arange(0, 4)

    labels = a.reshape(-1, 1)

    gen_imgs = generator.predict([z, labels])

    # gen_imgs = 0.5 * gen_imgs + 0.5
    print("Epoch : %d " % (epoch + 1))
    fig, axs = plt.subplots(rows, columns, )

    cnt = 0
    for i in range(rows):
        for j in range(columns):
            example = (gen_imgs[cnt] + 1) / 2.0
            axs[i, j].imshow(example)
            axs[i, j].axis('off')
            axs[i, j].set_title("Type: %s" % dict_clothes.get(labels[cnt][0]))
            cnt += 1
    fig.savefig('./exec_cgan/image%d.jpg' % (epoch))


# define training step
def train(epochs, batch_size, sample_interval):
    #  import Fashion-MNIST dataset
    X_train, Y_train = load_all_imgs('./dataset/dataset_64x64', classes, (64, 64))

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], Y_train[idx]

        z = np.random.normal(0, 1, (batch_size, z_dim))
        # generate images from generator
        gen_imgs = generator.predict([z, labels])
        # pass real an generated images to the discriminator and ctrain on them
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, z_dim))

        labels = np.random.randint(0, n_class, batch_size).reshape(-1, 1)

        g_loss = cgan.train_on_batch([z, labels], real)

        if (epoch + 1) % sample_interval == 0:
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch + 1, d_loss[0], 100 * d_loss[1], g_loss))

            plot_sample_images(epoch + 1)


iterations = 20000
batch_size = 128
sample_interval = 100

train(iterations, batch_size, sample_interval)
