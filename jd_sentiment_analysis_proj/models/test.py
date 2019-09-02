import matplotlib.pyplot as plt
import re


def fetch(file):
    with open(file, mode='r', encoding='utf-8') as f:
        all = f.read()
        items = re.finditer('loss: (.*?) - acc: (.*?) - val_loss: (.*?) - val_acc: (.*)', all)
        for item in items:
            yield (item.group(1), item.group(2), item.group(3), item.group(4))

        # items = re.findall(" — val_f1: 0.\\d+", all)
        # lst = []
        # for item in items:
        #     lst.append(float(re.sub(' — val_f1: ', '', item)))
        # return lst


if __name__ == '__main__':
    # lst_lstm = fetch('lstm_log.log')
    lst_att = fetch('attention_lstm.log')

    loss, acc, val_loss, val_acc = [], [], [], []
    for item in lst_att:
        loss.append(float(item[0]))
        acc.append(float(item[1]))
        val_loss.append(float(item[2]))
        val_acc.append(float(item[3]))

    plt.scatter(range(len(loss)), loss, label='loss')
    plt.scatter(range(len(acc)), acc, label='acc')
    plt.scatter(range(len(val_loss)), val_loss, label='val_loss')
    plt.scatter(range(len(val_acc)), val_acc, label='val_acc')

    # plt.plot(range(len(lst_lstm)), lst_lstm, c='r', label='lstm')
    # plt.plot(range(len(lst_att)), lst_att, c='b', label='attention')

    plt.legend(loc='best')
    plt.show()
