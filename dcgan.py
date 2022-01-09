_base_ = [
    '../_base_/models/dcgan/dcgan_128x128.py',
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/default_runtime.py'
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

# define dataset
# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=128,
    train=dict(imgs_root='data/dysplasia/train/', pipeline=train_pipeline))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000)
]

total_iters = 300002

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64)))abc
