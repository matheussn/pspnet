import numpy as np
from matplotlib import pyplot as plt

from utils.dataset import load_real_samples_cgan

(X_train, y_train, names) = load_real_samples_cgan(target_size=(64, 64))

idx = np.random.randint(0, X_train.shape[0], 16)
imgs, labels, names = X_train[idx], y_train[idx], names[idx]

fig, axs = plt.subplots(2, 2)

cnt = 0
for i in range(2):
    for j in range(2):
        # Output a grid of images
        examples = (imgs[cnt] + 1) / 2.0
        axs[i, j].imshow(examples)
        axs[i, j].axis('off')
        axs[i, j].set_title(f"{labels[cnt]}: {names[cnt]}")
        cnt += 1
filename = f'exec_cgan/generated_plot_e%03d.png' % (100 + 1)
plt.savefig(filename)
plt.close()
