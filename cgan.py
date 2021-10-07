import numpy as np
from matplotlib import pyplot as plt
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session

from cgan_model import build_cgan_discriminator, build_cgan_generator, build_cgan
from dataset_utils import load_all_imgs

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(Session(config=config))

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


def train(epochs, batch_size, sample_interval):
    classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    (X_train, y_train) = load_all_imgs('./dataset/dataset_64x64', classes, (64, 64))

    bat_per_epo = int(X_train.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

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
            labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

            # Train the Generator
            g_loss = cgan.train_on_batch([z, labels], real)
            print('>%d, %d/%d, [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch + 1, j + 1, bat_per_epo, d_loss[0], 100 * d_loss[1], g_loss))

        if (epoch + 1) % sample_interval == 0:
            # Output training progress

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # Output sample of generated images
            sample_images(epoch)

#
# def train(iterations, batch_size, sample_interval):
#     classes = {'healthy': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
#     (X_train, y_train) = load_all_imgs('./dataset/dataset_64x64', classes, (64, 64))
#
#     real = np.ones((batch_size, 1))
#     fake = np.zeros((batch_size, 1))
#
#     for iteration in range(iterations):
#
#         # -------------------------
#         #  Train the Discriminator
#         # -------------------------
#
#         # Get a random batch of real images and their labels
#         idx = np.random.randint(0, X_train.shape[0], batch_size)
#         imgs, labels = X_train[idx], y_train[idx]
#
#         # Generate a batch of fake images
#         z = np.random.normal(0, 1, (batch_size, z_dim))
#         gen_imgs = generator.predict([z, labels])
#
#         # Train the Discriminator
#         d_loss_real = discriminator.train_on_batch([imgs, labels], real)
#         d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#
#         # ---------------------
#         #  Train the Generator
#         # ---------------------
#
#         # Generate a batch of noise vectors
#         z = np.random.normal(0, 1, (batch_size, z_dim))
#
#         # Get a batch of random labels
#         labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
#
#         # Train the Generator
#         g_loss = cgan.train_on_batch([z, labels], real)
#
#         if (iteration + 1) % sample_interval == 0:
#             # Output training progress
#             print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
#                   (iteration + 1, d_loss[0], 100 * d_loss[1], g_loss))
#
#             # Save losses and accuracies so they can be plotted after training
#             losses.append((d_loss[0], g_loss))
#             accuracies.append(100 * d_loss[1])
#
#             # Output sample of generated images
#             sample_images(iteration)


def sample_images(epoch, image_grid_rows=2, image_grid_columns=2):
    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Get image labels 0-9
    labels = np.arange(0, 4).reshape(-1, 1)

    # Generate images from random noise
    gen_imgs = generator.predict([z, labels])

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, )

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Rescale image pixel values to [0, 1]
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
batch_size = 16
sample_interval = 10

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
