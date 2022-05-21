# Check Pytorch installation

import json
import os.path as osp

import matplotlib.pyplot as plt
import mmcv
from mmcv import Config
from mmseg.apis import train_segmentor
# Check MMSegmentation installation
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.models import build_segmentor

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Execute models.')
    parser.add_argument('-dir', metavar='directory', dest='out_dir', type=str, help='Output directory name.', required=True)
    parser.add_argument('-dataset', metavar='data_root', dest='dataset_dir', type=str, help='Dataset root directory name.', required=True)
    parser.add_argument('-log-interval', metavar='interval', dest='log_interval', type=int, required=False, default=10, help='Interval to log_config')

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
    filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.tif')]
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


    cfg = Config.fromfile('../models/pspnet.py')
    cfg.runner.max_epochs = 50
    cfg.work_dir = f'./work_dirs/{args.out_dir}'
    cfg.optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
    cfg.data_root = data_root
    cfg.data.train.data_root = data_root
    cfg.data.train.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(240, 448), ratio_range=(1, 1)),
        dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
    cfg.data.val.data_root = data_root
    cfg.data.test.data_root = data_root
    cfg.log_config = dict(interval=args.log_interval, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
    # print(f'Config:\n{cfg.pretty_text}')

    datasets = [build_dataset(cfg.data.train)]
    # Build the detector
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
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
    plt.plot([len(m_acc), len(m_acc), 0], [0, m_acc[-1], m_acc[-1]], 'k-', lw=1,dashes=[2, 2])
    plt.plot([len(m_acc), len(m_acc), 0], [0, loss[-1], loss[-1]], 'k-', lw=1,dashes=[2, 2])
    plt.savefig(f'./work_dirs/{args.out_dir}/Loss_vs_Acc_Graph.png')
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.plot(range(1, len(m_acc) + 1), m_dice, label="M Dice")
    plt.xticks(range(1, len(m_acc) + 1), rotation=90)
    plt.title("M Dice Graph")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage")
    plt.legend()
    plt.plot([len(m_dice), len(m_dice), 0], [0, m_dice[-1], m_dice[-1]], 'k-', lw=1,dashes=[2, 2])
    plt.savefig(f'./work_dirs/{args.out_dir}/M_Dice_Graph.png')
    plt.close()
