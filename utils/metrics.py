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
    accuracies = []
    epoch = []

    def add_accuracy(self, acc):
        self.accuracies.append(100 * acc)

    def add_d_loss(self, d_loss):
        self.d_losses.append(d_loss)

    def add_g_loss(self, g_loss):
        self.g_losses.append(g_loss)

    def add_epoch(self, epoch: int):
        self.epoch.append(epoch)

    def plot_losses(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch, self.d_losses, label="Discriminator loss")
        plt.plot(self.epoch, self.g_losses, label="Generator loss")

        plt.xticks(self.epoch, rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{base_dir}/losses.png')

    def plot_accuracy(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch, self.accuracies, label="Discriminator accuracy")

        plt.xticks(self.epoch, rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(f'{base_dir}/accuracy.png')
