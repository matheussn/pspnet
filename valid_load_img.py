import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_all_imgs(path, image_size, mode):
    imgs = []
    labels = []
    files = os.listdir(path)
    files.sort()
    print(f'\n{path} found {len(files)} img to load')
    i = 0
    for file in files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, mode)
        if mode == 0:
            img = np.expand_dims(img, 2)
        img = cv2.resize(img, image_size)
        img = img / float(255)
        imgs.append(img)
        labels.append(0)
        if i % 100 == 0:
            print(f'\n[{i}]:', end='')
        else:
            print("|", end='')
        i = i + 1
    return np.array(imgs), np.array(labels)


# (X_train, y_train) = load_real_samples_cgan(target_size=(64, 64))
(X_train, y_train) = load_all_imgs('./dataset/healthy_64x64/', image_size=(64, 64), mode=1)
idx = np.random.randint(0, X_train.shape[0], 16)

imgs, labels = X_train[idx], y_train[idx]

fig, axs = plt.subplots(2, 2)
cnt = 0
for i in range(2):
    for j in range(2):
        # Output a grid of images
        examples = (imgs[cnt] + 1) / 2.0
        axs[i, j].imshow(examples)
        axs[i, j].axis('off')
        axs[i, j].set_title(f"{labels[cnt]}")
        cnt += 1
filename = f'exec_cgan/generated_plot_e%03d.png' % (100 + 1)
plt.savefig(filename)

plt.close()
print(imgs[0])

print(imgs[0].shape)
