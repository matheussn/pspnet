import tensorflow as tf
from tensorflow import keras
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, BinaryAccuracy, Accuracy
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session

from utils.metrics_util import Metrics
from utils.train import save_plot2

"""
## Prepare CelebA data
We'll use face images from the CelebA dataset, resized to 64x64.
"""
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(Session(config=config))

# os.makedirs("celeba_gan")
#
# url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
# output = "celeba_gan/data.zip"
# gdown.download(url, output, quiet=True)
#
# with ZipFile("celeba_gan/data.zip", "r") as zipobj:
#     zipobj.extractall("celeba_gan")

"""
Create a dataset from our folder, and rescale the images to the [0-1] range:
"""

dataset = keras.preprocessing.image_dataset_from_directory(
    "dataset/", label_mode=None, image_size=(64, 64), batch_size=8,
)
dataset = dataset.map(lambda x: x / 255.0)

"""
Let's display a sample image:
"""

# for x in dataset:
#     plt.axis("off")
#     plt.imshow((x.numpy() * 255).astype("int32")[0])
#     break

"""
## Create the discriminator
It maps a 64x64 image to a binary classification score.
"""

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

"""
## Create the generator
It mirrors the discriminator, replacing `Conv2D` layers with `Conv2DTranspose` layers.
"""

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

"""
## Override `train_step`
"""


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")
        self.d_acc_real_metric = BinaryAccuracy(name="d_acc_real")
        self.d_acc_fake_metric = BinaryAccuracy(name="d_acc_fake")
        self.d_acc_test_metric = BinaryAccuracy(name="d_acc_t")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.d_acc_real_metric, self.d_acc_fake_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions1 = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions1)
            self.d_acc_real_metric.update_state(labels, predictions1)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
            self.d_acc_fake_metric.update_state(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        test = tf.concat([misleading_labels, labels], 0)
        test2 = tf.concat([predictions, predictions1], 0)

        self.d_acc_test_metric.update_state(test, test2)

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "d_acc_r": self.d_acc_real_metric.result(),
            "d_acc_f": self.d_acc_fake_metric.result(),
            "d_acc_t": self.d_acc_test_metric.result(),
        }


"""
## Create a callback that periodically saves generated images
"""


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        save_plot2(generated_images, epoch, 'exec')
        # for i in range(self.num_img):
        #     img = keras.preprocessing.image.array_to_img(generated_images[i])
        #     img.save("generated_img_%03d_%d.png" % (epoch, i))


"""
## Train the end-to-end model
"""

epochs = 300  # In practice, use ~100 epochs

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=2e-4),
    loss_fn=BinaryCrossentropy()
)

history = gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(num_img=49, latent_dim=latent_dim)])

Metrics().add_g_loss(history.history['g_loss'])
Metrics().add_d_loss(history.history['d_loss'])
Metrics().add_fake_accuracy(history.history['d_acc_f'])
Metrics().add_real_accuracy(history.history['d_acc_r'])
Metrics().add_test_accuracy(history.history['d_acc_t'])
Metrics().epoch = [x for x in range(len(history.history["d_loss"]))]
Metrics().plot_losses('exec')
Metrics().plot_accuracy('exec')

"""
Some of the last generated images around epoch 30
(results keep improving after that):
![results](https://i.imgur.com/h5MtQZ7l.png)
"""
