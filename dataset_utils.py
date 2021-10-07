import os
import re

import cv2
import numpy as np


def bgr_to_rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


def get_label(file: str, labels: dict):
    match = re.search('[a-z]+', file).group(0)
    return labels[match]


def load_all_imgs(path, dict_label, img_size):
    imgs = []
    labels = []
    files = os.listdir(path)
    files.sort()
    print(f'\n{path} found {len(files)} img to load')
    for index, file in enumerate(files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = bgr_to_rgb(img)
        img = cv2.resize(img, img_size)
        img = (img / 127.5) - 1.
        imgs.append(img)
        labels.append(get_label(file, dict_label))
        # if index % 100 == 0:
        #     print(f'\n[{index}]:', end='')
        # else:
        #     print("|", end='')
    return np.array(imgs), np.array(labels)
