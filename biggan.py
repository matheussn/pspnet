_base_ = [
    'mmgeneration/configs/_base_/models/biggan/biggan_128x128.py',
    './displasia_128.py', 
    'mmgeneration/configs/_base_/default_runtime.py'
]

# define dataset
# you must set `samples_per_gpu`
# data = dict(samples_per_gpu=25, workers_per_gpu=8)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(data_prefix='data/dysplasia'),
    val=dict(data_prefix='data/dysplasia'))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=500),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=4,
        start_iter=400,
        interp_cfg=dict(momentum=0.9999),
        priority='VERY_HIGH')
]

total_iters = 5000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

# Note set your inception_pkl's path
inception_pkl = None
evaluation = dict(
    type='GenerativeEvalHook',
    interval=1000,
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            inception_pkl=inception_pkl,
            bgr2rgb=True),
        dict(type='IS', num_images=50000)
    ],
    sample_kwargs=dict(sample_model='ema'),
    best_metric=['fid', 'is'])

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        bgr2rgb=True),
    is50k=dict(type='IS', num_images=50000))

work_dir = './work_dirs/tutorial'
  
seed = 0
gpu_ids = range(1)