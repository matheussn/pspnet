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

    def add_real_accuracy_one(self, acc):
        self.accuracies_real.append(100 * acc)

    def add_test_accuracy(self, acc):
        self.accuracies_test.append([100 * x for x in acc])

    def add_test_accuracy_one(self, acc):
        self.accuracies_test.append(100 * acc)

    def add_fake_accuracy(self, acc):
        self.accuracies_fake.append([100 * x for x in acc])

    def add_fake_accuracy_one(self, acc):
        self.accuracies_fake.append(100 * acc)

    def add_d_loss(self, d_loss):
        self.d_losses.append(100 * d_loss)

    def add_g_loss(self, g_loss):
        self.g_losses.append(100 * g_loss)

    def add_epoch(self, epoch: int):
        self.epoch.append(epoch)

    def _get_100_plots(self, n: list):
        index = [x for x in range(0, len(n), 100)]
        return [n[i] for i in index]

    def _get_20_plots(self, n: list):
        index = [x for x in range(0, len(n), 20)]
        return [n[i] for i in index]

    def _get_1000_plots(self, n: list):
        index = [x for x in range(0, len(n), 1000)]
        return [n[i] for i in index]

    def _plot_100_loss(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self._get_100_plots(self.epoch), self._get_100_plots(self.d_losses), label="Discriminator loss")
        plt.plot(self._get_100_plots(self.epoch), self._get_100_plots(self.g_losses), label="Generator loss")

        plt.xticks(self._get_100_plots(self.epoch), rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{base_dir}/losses_100.png')
        plt.close()

    def _plot_1000_loss(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self._get_1000_plots(self.epoch), self._get_1000_plots(self.d_losses), label="Discriminator loss")
        plt.plot(self._get_1000_plots(self.epoch), self._get_1000_plots(self.g_losses), label="Generator loss")

        plt.xticks(self._get_1000_plots(self.epoch), rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{base_dir}/losses_1000.png')
        plt.close()

    def _plot_20_loss(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self._get_20_plots(self.epoch), self._get_20_plots(self.d_losses), label="Discriminator loss")
        plt.plot(self._get_20_plots(self.epoch), self._get_20_plots(self.g_losses), label="Generator loss")

        plt.xticks(self._get_20_plots(self.epoch), rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{base_dir}/losses_20.png')
        plt.close()

    def plot_losses(self, base_dir, epochs):
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch, self.d_losses, label="Discriminator loss")
        plt.plot(self.epoch, self.g_losses, label="Generator loss")

        # plt.xticks(self.epoch, rotation=90)

        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{base_dir}/losses.png')
        plt.close()

        self._plot_20_loss(base_dir)
        self._plot_100_loss(base_dir)
        if epochs >= 1000:
            self._plot_1000_loss(base_dir)

    def _plot_20_accuracy(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self._get_20_plots(self.epoch), self._get_20_plots(self.accuracies_fake),
                 label="Discriminator accuracy (Fake)")
        plt.plot(self._get_20_plots(self.epoch), self._get_20_plots(self.accuracies_real),
                 label="Discriminator accuracy (Real)")
        plt.plot(self._get_20_plots(self.epoch), self._get_20_plots(self.accuracies_test),
                 label="Discriminator accuracy (test)")

        plt.xticks(self._get_20_plots(self.epoch), rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(f'{base_dir}/accuracy_20.png')
        plt.close()

    def _plot_100_accuracy(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self._get_100_plots(self.epoch), self._get_100_plots(self.accuracies_fake),
                 label="Discriminator accuracy (Fake)")
        plt.plot(self._get_100_plots(self.epoch), self._get_100_plots(self.accuracies_real),
                 label="Discriminator accuracy (Real)")
        plt.plot(self._get_100_plots(self.epoch), self._get_100_plots(self.accuracies_test),
                 label="Discriminator accuracy (test)")

        plt.xticks(self._get_100_plots(self.epoch), rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(f'{base_dir}/accuracy_100.png')
        plt.close()

    def _plot_1000_accuracy(self, base_dir):
        plt.figure(figsize=(15, 5))
        plt.plot(self._get_1000_plots(self.epoch), self._get_1000_plots(self.accuracies_fake),
                 label="Discriminator accuracy (Fake)")
        plt.plot(self._get_1000_plots(self.epoch), self._get_1000_plots(self.accuracies_real),
                 label="Discriminator accuracy (Real)")
        plt.plot(self._get_1000_plots(self.epoch), self._get_1000_plots(self.accuracies_test),
                 label="Discriminator accuracy (test)")

        plt.xticks(self._get_1000_plots(self.epoch), rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(f'{base_dir}/accuracy_1000.png')
        plt.close()

    def plot_accuracy(self, base_dir, epochs):
        plt.figure(figsize=(15, 5))
        plt.plot(self.epoch, self.accuracies_fake, label="Discriminator accuracy (Fake)")
        plt.plot(self.epoch, self.accuracies_real, label="Discriminator accuracy (Real)")
        plt.plot(self.epoch, self.accuracies_test, label="Discriminator accuracy (test)")

        plt.xticks(self.epoch, rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Discriminator Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(f'{base_dir}/accuracy.png')
        plt.close()

        self._plot_20_accuracy(base_dir)
        self._plot_100_accuracy(base_dir)
        if epochs >= 1000:
            self._plot_1000_accuracy(base_dir)
