import matplotlib.pyplot as plt
import numpy as np


def acc_loss_draw(epochs, datalist, legend_name, titlename):
    epochs_list = [i + 1 for i in range(epochs)]
    for i in range(len(datalist)):
        plt.plot(epochs_list, datalist[i], label=legend_name[i])
    plt.xlabel("epoch")
    plt.legend()
    plt.title(titlename)
    plt.show()
