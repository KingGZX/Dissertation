import matplotlib.pyplot as plt
import numpy as np
from config import Config

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
]

scatter_style = [
    '.',
    ',',
    'o',
    'v',
    '^',
    '<',
    '>',
    '8',
    's',
    'p',
    '*',
    '+',
    'D',
    'd',
    'x',
    '|',
    '_'
]


def item_acc_loss_draw(epochs, data_list, legend_name, title_name, save_path: str, item_name: str):
    epochs_list = [i + 1 for i in range(epochs)]
    for i in range(len(data_list)):
        plt.plot(epochs_list, data_list[i], label=legend_name[i])
    plt.xlabel("epoch")
    plt.legend()
    plt.title(title_name)
    # plt.show()
    plt.savefig(save_path + "/" + item_name + ".png")
    plt.clf()


def total_loss_draw(epochs, loss_list, legend_name, title_name, save_path: str, name: str):
    epochs_list = [i + 1 for i in range(epochs)]
    plt.plot(epochs_list, loss_list, label=legend_name)
    plt.xlabel("epoch")
    plt.legend()
    plt.title(title_name)
    # plt.show()
    plt.savefig(save_path + "/" + name + ".png")
    plt.clf()


def compare_item_acc_draw(epochs, item_loss_list, legend_name, title_name, save_path: str, item_name: str):
    epochs_list = [i + 1 for i in range(epochs)]
    for i in range(len(item_loss_list)):
        plt.scatter(epochs_list, item_loss_list[i], marker=scatter_style[i])
        plt.plot(epochs_list, item_loss_list[i], label=legend_name[i])
    plt.xlabel("epoch")
    plt.legend()
    plt.title(title_name)
    # plt.show()
    plt.savefig(save_path + "/" + item_name + ".png")
    plt.clf()
