import matplotlib.pyplot as plt
import numpy as np
from config import Config


def acc_loss_draw(epochs, data_list, legend_name, title_name):
    epochs_list = [i + 1 for i in range(epochs)]
    for i in range(len(data_list)):
        plt.plot(epochs_list, data_list[i], label=legend_name[i])
    plt.xlabel("epoch")
    plt.legend()
    plt.title(title_name)
    plt.show()


def items_acc_acc_draw(epochs, items_acc_list, cfg: Config):
    epochs_list = [i + 1 for i in range(epochs)]
    for item in range(len(cfg.item)):
        item_acc_list = [items_acc_list[i][item] for i in range(epochs)]
        plt.plot(epochs_list, item_acc_list)
        plt.xlabel("epoch")
        plt.title(cfg.label_dict["item" + str(cfg.item[item])])
        plt.show()
