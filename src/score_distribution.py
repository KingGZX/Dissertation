from config import Config
import matplotlib.pyplot as plt
import numpy as np

# For Dr.Patrick Kwong

"""
图片显示后 自行放大到全图
并且利用!!!!窗口自带!!!!（无需代码控制）的调整工具调整一下 vspace 控制子图的间距
"""

if __name__ == "__main__":
    cfg = Config()
    labels = cfg.labels  # Dataframe

    fig = plt.figure()
    fig.suptitle("score distribution of each item")

    for i in range(len(cfg.classes)):
        item = labels.iloc[:, i + 2]
        scores = [int(item[t][-1]) for t in range(len(item))]
        x_labels = [j + 1 for j in range(cfg.classes[i])]
        counts = list()
        for x in x_labels:
            count = 0
            for score in scores:
                if score == x:
                    count += 1
            counts.append(count)
        plt.subplot(4, 4, i + 1)
        plt.title(cfg.label_dict["item" + str(i + 1)], font={'family': 'Arial', 'size': 8})
        plt.bar(x_labels, counts, width=0.5)

    plt.show()
