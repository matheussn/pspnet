import numpy as np
from matplotlib import pyplot
from numpy.random import randint, randn

from utils.metrics import Metrics

LOG_FILE = 0


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return x, y


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return x, y


def save_plot(examples, epoch, base_dir, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = f'{base_dir}/generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, base_dir, n_samples=150):
    # prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    loss_real, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    loss_fake, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    d_loss = 0.5 * np.add([loss_real, acc_real], [loss_fake, acc_fake])
    with open(f'{base_dir}/logs.txt', 'a') as log:
        print('>Accuracy real: %.0f%%, fake: %.0f%%, acc.: %.0f%%' % (acc_real * 100, acc_fake * 100, 100 * d_loss[1]),
              file=log)
    # save plot
    save_plot(x_fake, epoch, base_dir)

    Metrics().add_d_loss(d_loss[0])
    Metrics().add_accuracy(d_loss[1])
    Metrics().add_epoch(epoch + 1)
    # save the generator model tile file
    filename = f'{base_dir}/generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train(g_model, d_model, gan_model, dataset, base_dir, latent_dim=100, n_epochs=300, n_batch=8):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for epoch in range(n_epochs):
        # enumerate batches over the training set
        g_loss = 0
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            x_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch(x_real, y_real)
            # generate 'fake' examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_real, _ = d_model.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            # summarize loss on this batch
            with open(f'{base_dir}/logs.txt', 'a') as log:
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (epoch + 1, j + 1, bat_per_epo, d_loss_fake, d_loss_real, g_loss), file=log)
        if (epoch + 1) % 10 == 0:
            # Save losses and accuracies so they can be plotted after training
            Metrics().add_g_loss(g_loss)
            summarize_performance(epoch, g_model, d_model, dataset, latent_dim, base_dir)

    Metrics().plot_accuracy(base_dir)
    Metrics().plot_losses(base_dir)
