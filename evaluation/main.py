import argparse

import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', required=True)

    args = parser.parse_args()

    file = open(args.file)
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

    plt.figure(figsize=(15, 5))
    plt.plot(range(1, len(m_acc) + 1), loss, label="loss")
    plt.plot(range(1, len(m_acc) + 1), m_acc, label="M Acc")
    plt.xticks(range(1, len(m_acc) + 1), rotation=90)
    plt.title("Loss vs Acc")
    plt.xlabel("Iteration")
    plt.ylabel("percentage")
    plt.legend()
    plt.savefig('Loss_vs_Acc_Graph.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(range(1, len(m_acc) + 1), m_dice, label="M Dice")
    plt.xticks(range(1, len(m_acc) + 1), rotation=90)
    plt.title("M Dice Graph")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage")
    plt.legend()
    plt.savefig('M_Dice_Graph.png')
    plt.show()
    plt.close()
