from matplotlib import pyplot as plt


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Metrics(metaclass=SingletonMeta):
    g_losses = []
    d_losses = []
    accuracies_fake = []
    accuracies_real = []
    accuracies_test = []
    epoch = []

    def add_real_accuracy(self, acc):
        self.accuracies_real.append([100 * x for x in acc])

    def add_test_accuracy(self, acc):
        self.accuracies_test.append([100 * x for x in acc])

    def add_fake_accuracy(self, acc):
        self.accuracies_fake.append([100 * x for x in acc])

    def add_d_loss(self, d_loss):
        self.d_losses.append(d_loss)

    def add_g_loss(self, g_loss):
        self.g_losses.append(g_loss)

    def add_epoch(self, epoch: int):
        self.epoch.append(epoch)

    def plot_losses(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch, self.d_losses[0], label="Discriminator loss")
        plt.plot(self.epoch, self.g_losses[0], label="Generator loss")

        plt.xticks(self.epoch, rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{base_dir}/losses.png')

    def plot_accuracy(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch, self.accuracies_fake[0], label="Discriminator accuracy (Fake)")
        plt.plot(self.epoch, self.accuracies_real[0], label="Discriminator accuracy (Real)")
        plt.plot(self.epoch, self.accuracies_test[0], label="Discriminator accuracy (test)")

        plt.xticks(self.epoch, rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(f'{base_dir}/accuracy.png')
