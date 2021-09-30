import numpy as np
from matplotlib import pyplot as plt
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LeakyReLU, Flatten, Dense, Conv2DTranspose, Reshape, Activation, Multiply, \
    Embedding, Conv2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session

from utils.dataset import load_real_samples_cgan

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(Session(config=config))


def build_generator(z_dim):
    model = Sequential(name="Generator")

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * 4 * 4, input_dim=z_dim))
    model.add(Reshape((4, 4, 256)))

    # Transposed convolution layer, from 4x4x256 into 8x8x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 8x8x128 to 8x8x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 8x8x128 to 16x16x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 16x16x128 to 32x32x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 32x32x128 to 32x32x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 32x32x64 to 64x64x3 tensor
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))

    return model


def build_cgan_generator(z_dim, num_classes):
    # Random noise vector z
    z = Input(shape=(z_dim,))
    # Conditioning label: integer 0-9 specifying the number G should generate
    label = Input(shape=(1,), dtype='int32')

    # Label embedding:
    # ----------------
    # Turns labels into dense vectors of size z_dim
    # Produces 3D tensor with shape (batch_size, 1, z_dim)
    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)

    # Flatten the embedding 3D tensor into 2D tensor with shape (batch_size, z_dim)
    label_embedding = Flatten()(label_embedding)

    # Element-wise product of the vectors z and the label embeddings
    joined_representation = Multiply()([z, label_embedding])

    generator = build_generator(z_dim)

    # Generate image for the given label
    conditioned_img = generator(joined_representation)

    return Model([z, label], conditioned_img)


def build_discriminator(img_shape):
    model = Sequential(name="discriminator")

    # Convolutional layer, from 64x64x3 into 32x32x64 tensor
    model.add(Conv2D(64, kernel_size=3, strides=2, activation="softmax",
                     input_shape=(img_shape[0], img_shape[1], img_shape[2] + 3),
                     padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 32x32x64 into 16x16x64 tensor
    model.add(Conv2D(64, kernel_size=3, activation="softmax", strides=2, input_shape=img_shape, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 16x16x64 tensor into 8x8x128 tensor
    model.add(Conv2D(128, kernel_size=3, activation="softmax", strides=2, input_shape=img_shape, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 8x8x64 tensor into 4x4x128 tensor
    model.add(Conv2D(128, kernel_size=3, activation="softmax", strides=2, input_shape=img_shape, padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='softmax'))

    return model


def build_cgan_discriminator(img_shape, num_classes):
    # Input image
    img = Input(shape=img_shape)

    # Label for the input image
    label = Input(shape=(1,), dtype='int32')

    # Label embedding:
    # ----------------
    # Turns labels into dense vectors of size z_dim
    # Produces 3D tensor with shape (batch_size, 1, 28*28*1)
    label_embedding = Embedding(num_classes, np.prod(img_shape), input_length=1)(label)

    # Flatten the embedding 3D tensor into 2D tensor with shape (batch_size, 28*28*1)
    label_embedding = Flatten()(label_embedding)

    # Reshape label embeddings to have same dimensions as input images
    label_embedding = Reshape(img_shape)(label_embedding)

    # Concatenate images with their label embeddings
    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = build_discriminator(img_shape)

    # Classify the image-label pair
    classification = discriminator(concatenated)

    return Model([img, label], classification)


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


img_rows = 64
img_cols = 64
channels = 3

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100

# Number of classes in the dataset
num_classes = 4

# Build and compile the Discriminator
discriminator = build_cgan_discriminator(img_shape, num_classes)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Build the Generator
generator = build_cgan_generator(z_dim, num_classes)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

# Build and compile CGAN model with fixed Discriminator to train the Generator
cgan = build_cgan(generator, discriminator, z_dim)
cgan.compile(loss='binary_crossentropy', optimizer=Adam())

# discriminator.summary()
# generator.get_layer("Generator").summary()
# discriminator.get_layer("discriminator").summary()

accuracies = []
losses = []


def train(iterations, batch_size, sample_interval):
    (X_train, y_train) = load_real_samples_cgan(target_size=(img_rows, img_cols))

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    # X_train = X_train / 127.5 - 1.
    # X_train = np.expand_dims(X_train, axis=3)

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get a random batch of real images and their labels
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict([z, labels])

        # Train the Discriminator
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of noise vectors
        z = np.random.normal(0, 1, (batch_size, z_dim))

        # Get a batch of random labels
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        # Train the Generator
        g_loss = cgan.train_on_batch([z, labels], real)

        if (iteration + 1) % sample_interval == 0:
            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss))

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # Output sample of generated images
            sample_images(iteration)


def sample_images(epoch, image_grid_rows=2, image_grid_columns=2):
    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Get image labels 0-9
    labels = np.arange(0, 4).reshape(-1, 1)

    # Generate images from random noise
    gen_imgs = generator.predict([z, labels])

    # Rescale image pixel values to [0, 1]
    # gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            img = (gen_imgs[cnt] + 1) / 2.0
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" % labels[cnt])
            cnt += 1
    filename = f'exec_cgan/generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


# Set hyperparameters
iterations = 12000
batch_size = 128
sample_interval = 1000

# Train the CGAN for the specified number of iterations
train(iterations, batch_size, sample_interval)

epoch = [x for x in range(0, iterations, 1000)]

plt.figure(figsize=(15, 5))
plt.plot(epoch, accuracies, label="Discriminator accuracy")

plt.xticks(epoch, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig(f'exec_cgan/accuracy.png')
plt.close()

plt.figure(figsize=(15, 5))
plt.plot(epoch, [x[0] for x in losses], label="Discriminator loss")
plt.plot(epoch, [x[1] for x in losses], label="Generator loss")

# plt.xticks(self.epoch, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'exec_cgan/losses.png')
plt.close()
