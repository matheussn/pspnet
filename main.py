# Check Pytorch installation
import argparse
import json
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mmcv
import torch
from mmcv import Config
from mmseg.apis import train_segmentor
# Check MMSegmentation installation
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.models import build_segmentor
from glob import glob
import os
import cv2 as cv
from mmseg.apis import inference_segmentor
import sys

if __name__ == '__main__':

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Execute models.')
    parser.add_argument('-dir', metavar='directory', dest='out_dir', type=str, help='Output directory name.',
                        required=True)
    parser.add_argument('-dataset', metavar='data_root', dest='dataset_dir', type=str,
                        help='Dataset root directory name.', required=True)
    parser.add_argument('-log-interval', metavar='interval', dest='log_interval', type=int, required=False, default=10,
                        help='Interval to log_config')
    parser.add_argument('-model', metavar='model', dest='model', type=str, required=False, default='pspnet.py',
                        help='Model to run')
    parser.add_argument('-epochs', metavar='epochs', dest='epochs', type=int, required=False, default=50,
                        help='Number of epochs to run the model.')
    parser.add_argument('-opt', metavar='optimizer', dest='optimizer', type=str, required=False, default='SGD')

    args = parser.parse_args()

    if osp.isdir(f'./work_dirs/{args.out_dir}'):
        print(f'Directory with name {args.out_dir} already exists.')
        print('Choice another name to output directory.')
        exit(-1)

    if not osp.isdir(args.dataset_dir):
        print(f'Directory with name {args.dataset_dir} not found!')
        exit(-1)

    data_root = args.dataset_dir
    img_dir = 'images'
    ann_dir = 'annotations'

    classes = ('bg', 'cell')
    palette = [[0, 0, 0], [255, 255, 255]]

    # split train/val set randomly
    split_dir = 'splits'
    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in
                     mmcv.scandir(osp.join(data_root, ann_dir), suffix='.tif')]
    with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
        train_length = int(len(filename_list) * 4 / 5)
        f.writelines(line + '\n' for line in filename_list[:train_length])
    with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
        f.writelines(line + '\n' for line in filename_list[train_length:])


    @DATASETS.register_module()
    class DysplasiaDataSet(CustomDataset):
        CLASSES = classes
        PALETTE = palette

        def __init__(self, split, **kwargs):
            super().__init__(img_suffix='.tif', seg_map_suffix='.tif', split=split, **kwargs)
            assert osp.exists(self.img_dir) and self.split is not None


    cfg = Config.fromfile(f'../models/{args.model}')
    cfg.runner.max_epochs = args.epochs
    cfg.work_dir = f'./work_dirs/{args.out_dir}'
    cfg.data_root = data_root
    cfg.data.train.data_root = data_root
    cfg.data.val.data_root = data_root
    cfg.data.test.data_root = data_root
    cfg.log_config = dict(interval=args.log_interval, hooks=[dict(type='TextLoggerHook', by_epoch=True)])

    if args.optimizer == 'Adam':
        cfg.optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)

    # print(f'Config:\n{cfg.pretty_text}')

    datasets = [build_dataset(cfg.data.train)]
    # Build the detector
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    with open(f'./work_dirs/{args.out_dir}/config.txt', 'w') as config:
        config.write(f'Command: {sys.argv} \n Config: \n {cfg.pretty_text}')

    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())

    file = open(f'./work_dirs/{args.out_dir}/None.log.json')
    lines = file.readlines()

    m_dice = []
    m_acc = []
    loss = []

    for line in lines:
        json_line = json.loads(line)
        if len(json_line):
            if json_line['mode'] == 'train':
                loss.append(json_line['loss'])
            else:
                m_acc.append(json_line['mAcc'])
                m_dice.append(json_line['mDice'])

    plt.figure(figsize=(20, 5))
    plt.plot(range(1, len(m_acc) + 1), loss, label="loss")
    plt.plot(range(1, len(m_acc) + 1), m_acc, label="M Acc")
    plt.xticks(range(1, len(m_acc) + 1), rotation=90)
    plt.title("Loss vs Acc")
    plt.xlabel("Iteration")
    plt.ylabel("percentage")
    plt.legend()
    plt.plot([len(m_acc), len(m_acc), 0], [0, m_acc[-1], m_acc[-1]], 'k-', lw=1, dashes=[2, 2])
    plt.plot([len(m_acc), len(m_acc), 0], [0, loss[-1], loss[-1]], 'k-', lw=1, dashes=[2, 2])
    plt.savefig(f'./work_dirs/{args.out_dir}/Loss_vs_Acc_Graph.png')
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.plot(range(1, len(m_acc) + 1), m_dice, label="M Dice")
    plt.xticks(range(1, len(m_acc) + 1), rotation=90)
    plt.title("M Dice Graph")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage")
    plt.legend()
    plt.plot([len(m_dice), len(m_dice), 0], [0, m_dice[-1], m_dice[-1]], 'k-', lw=1, dashes=[2, 2])
    plt.savefig(f'./work_dirs/{args.out_dir}/M_Dice_Graph.png')
    plt.close()

    last_iteration = {"acc": m_acc[-1], "loss": loss[-1], "dice": m_dice[-1], "iter": len(m_acc)}
    baIndex = m_acc.index(np.amax(m_acc))
    best_acc = {"acc": m_acc[baIndex], "loss": loss[baIndex], "dice": m_dice[baIndex], "iter": baIndex + 1}
    blIndex = loss.index(np.amin(loss))
    best_loss = {"acc": m_acc[blIndex], "loss": loss[blIndex], "dice": m_dice[blIndex], "iter": blIndex + 1}
    bdIndex = m_dice.index(np.amax(m_dice))
    best_dice = {"acc": m_acc[bdIndex], "loss": loss[bdIndex], "dice": m_dice[bdIndex], "iter": bdIndex + 1}

    metrics = ["Acc", "Loss", "Dice"]
    val2 = [
        f"Last Iteration ({last_iteration['iter']})",
        f"Best accurracy ({best_acc['iter']})",
        f"Best Loss ({best_loss['iter']})",
        f"Best dice ({best_dice['iter']})"
    ]
    val3 = [
        [last_iteration["acc"], last_iteration["loss"], last_iteration["dice"]],
        [best_acc["acc"], best_acc["loss"], best_acc["dice"]],
        [best_loss["acc"], best_loss["loss"], best_loss["dice"]],
        [best_dice["acc"], best_dice["loss"], best_dice["dice"]]
    ]

    # fig, ax = plt.subplots()
    # ax.set_axis_off()
    table = plt.table(
        cellText=val3,
        rowLabels=val2,
        colLabels=metrics,
        cellLoc='center',
        loc='center')
    plt.axis('off')
    plt.grid('off')
    plt.gcf().canvas.draw()
    points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
    points[0, :] -= 10
    points[1, :] += 10
    nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)

    plt.savefig(f"./work_dirs/{args.out_dir}/metrics.png", bbox_inches=nbbox)
    plt.close()

    images_folder = '../data_aug/ToTrain/images'
    train_images = []
    for file in os.listdir(images_folder):
        img_path = os.path.join(images_folder, file)
        img = mmcv.imread(img_path)
        train_images.append((img, f'./work_dirs/{args.out_dir}/raw_res/' + img_path[27:]))

    mmcv.mkdir_or_exist(osp.abspath(f'{cfg.work_dir}/raw_res'))
    opacity = 0.5
    fig_size=(15, 10)
    for img, path in train_images:
        model.cfg = cfg
        result = inference_segmentor(model, img)
        predict_img = result[0]
        predict_img[predict_img == 1] = 255
        predict_img[predict_img == 0] = 0
        cv.imwrite(f'{path}', predict_img)

