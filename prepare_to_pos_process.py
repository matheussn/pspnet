import os
import re

if __name__ == '__main__':
    result_path = 'results'
    raw_path = 'raw_res'

    model_paths = [
            'new_pspnet_resnest_50_ADAM',
            'new_pspnet_resnest_50_SGD',
            'new_pspnet_resnet_50_ADAM',
            'new_pspnet_resnet_50_SGD',
            'new_pspnet_unet_50_ADAM',
            'new_pspnet_unet_50_SGD'
            ]

    for model in model_paths:
        to_pos_path = os.path.join(result_path, model, 'to_pos')
        if not os.path.exists(to_pos_path):
            os.mkdir(to_pos_path)

        base_raw_path = os.path.join(result_path, model, raw_path)
        for image in os.listdir(base_raw_path):
            result = re.split("_", image)
            match = result[0]

            level_path = os.path.join(to_pos_path, match)

            if not os.path.exists(level_path):
                os.mkdir(level_path)

            os.popen(f'cp {os.path.join(base_raw_path, image)} {os.path.join(level_path, image[len(match)+1:])}')


