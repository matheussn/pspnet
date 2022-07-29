import torch
import json
from matplotlib import pyplot as plt

models = {'pspnet_unet_100_ADAM': 'Unet', 'pspnet_resnet_100_ADAM': 'resnet', 'pspnet_resnest_100_ADAM': 'resnest'}

metrics = {}

if __name__ == '__main__':

    torch.cuda.empty_cache()

    for model in models.keys():
        file = open(f'./{model}/None.log.json')
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

        metrics[model] = {
            'acc': m_acc,
            'dice': m_dice,
            'loss': loss,
        }

    plt.figure(figsize=(20, 5))
    for model in models.keys():
        acc = metrics[model]['acc']
        name = models[model]
        plt.plot(range(1, len(acc) + 1), acc, label=name)
        plt.xticks(range(1, len(acc) + 1), rotation=90)
    plt.title("")
    plt.xlabel("Número de épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.savefig(f'./metrics/all_acc.png')
    plt.close()

    plt.figure(figsize=(20, 5))
    for model in models.keys():
        acc = metrics[model]['loss']
        name = models[model]
        plt.plot(range(1, len(acc) + 1), acc, label=name)
        plt.xticks(range(1, len(acc) + 1), rotation=90)
    plt.title("")
    plt.xlabel("Número de épocas")
    plt.ylabel("Taxa de perda")
    plt.legend()
    plt.savefig(f'./metrics/all_loss.png')
    plt.close()

    plt.figure(figsize=(20, 5))
    for model in models.keys():
        acc = metrics[model]['dice']
        name = models[model]
        plt.plot(range(1, len(acc) + 1), acc, label=name)
        plt.xticks(range(1, len(acc) + 1), rotation=90)
    plt.title("")
    plt.xlabel("Número de épocas")
    plt.ylabel("Dice")
    plt.legend()
    plt.savefig(f'./metrics/all_dice.png')
    plt.close()
