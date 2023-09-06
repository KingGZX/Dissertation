import torch
import torch.optim as optim
import torch.nn as nn
from utils.plot import *
from loadata import Config
from utils.log import get_logger
from loadata import Dataset
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def draw(train_acc, test_acc, item_loss, total_loss, model: str, cfg: Config, epochs):
    # draw train accuracy, test accuracy and loss on the same figure
    import os
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    idx = len(os.listdir("./Results")) + 1
    fp = "./Results/exp" + str(idx)
    os.mkdir(fp)
    items = len(cfg.item)
    items_test_acc_list = list()
    item_names = list()
    for item in range(items):
        item_train, item_test, item_los = list(), list(), list()
        for epoch in range(epochs):
            item_train.append(train_acc[epoch][item])
            item_test.append(test_acc[epoch][item])
            item_los.append(item_loss[epoch][item])
        data_list = [item_train, item_test, item_los]
        legend_name = ["train accuracy", "test accuracy", "loss value"]
        item_name = cfg.label_dict["item" + str(cfg.item[item])]
        items_test_acc_list.append(item_test)
        item_names.append(item_name)
        title = model + "'s performance on item " + item_name
        item_acc_loss_draw(epochs, data_list=data_list, legend_name=legend_name,
                           title_name=title, save_path=fp, item_name="item" + str(cfg.item[item]))

    # draw a single picture of total loss trend
    if items > 1:
        total_loss_draw(epochs, loss_list=total_loss, legend_name="total loss",
                        title_name="total loss of model " + model, save_path=fp, name="total_loss")

    # make comparison of different items
    if items > 1:
        compare_item_acc_draw(epochs, items_test_acc_list, item_names, "test accuracy of each item", fp, "compare")


def batch_train(model, dataset: Dataset, epochs: int, model_name: str, cfg: Config, log=False, log_file=None):
    """
    :param model:        Network Instance
    :param dataset:      The class implemented in loadata.py
    :param epochs:       Hyperparameter
    :param model_name:   e.g. ST-GCN, Uniformer, GFN, for plotting result
    :param cfg:          config file
    :param log_file:     Logging file name
    :param log:          Logging setting
    :return:
    """
    # Hyperparameter setting
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    batch_size = cfg.batch_size
    train_len = len(dataset.train_data)
    batches = math.ceil(train_len / batch_size)

    # move the model to GPU
    model.to(device)

    # create log file handler
    if log:
        import os
        if not os.path.exists("./log"):
            os.mkdir("./log")
        idx = len(os.listdir("./log")) + 1
        fp = "exp" + str(idx) + ".log" if log_file is None else log_file
        logger = get_logger(
            filename="./log/" + fp,
            verbosity=1
        )

    item_loss_list, item_acc_train_list, total_loss_list, item_acc_test_list = list(), list(), list(), list()
    for epoch in range(epochs):
        # shuffle gait cycles in the train set
        dataset.shuffle()
        item_acc = np.zeros(len(cfg.item))
        item_loss = np.zeros(len(cfg.item))
        total_loss = 0
        for batch in range(batches):
            if batch_size > 1:
                train_data, train_label = dataset.load_batch_data_train()
            else:
                train_data, train_label, patient_name, gait_cycle = dataset.load_data(train=True)
            # to tensor
            train_data = torch.tensor(train_data, dtype=torch.float32).to(device, dtype=dtype)
            for i in range(len(cfg.item)):
                # each element of train_label is a list, it's corresponding labels of this batch data of item[i]
                train_label[i] = torch.tensor(train_label[i], dtype=torch.long).to(device)

            out = model(train_data)
            loss = torch.tensor(0).to(device, dtype=dtype)
            for i in range(len(cfg.item)):
                i_loss = criterion(out[i], train_label[i])
                loss += i_loss
                correct_num = torch.sum(torch.argmax(out[i], dim=1) == train_label[i]).cpu()
                item_acc[i] += correct_num
                item_loss[i] += i_loss.item()

            total_loss += loss.item()

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        item_acc = item_acc / train_len
        item_loss = item_loss / batches
        total_loss = total_loss / batches
        item_acc_train_list.append(item_acc), item_loss_list.append(item_loss), total_loss_list.append(total_loss)
        item_acc_test = validation(model, dataset, cfg)
        item_acc_test_list.append(item_acc_test)
        print("Epoch:{}/{}, average loss is {}".format(epoch, epochs, total_loss))
        for i in range(len(cfg.item)):
            print("Epoch:{}/{}, accuracy of item {} is {}".format(epoch, epochs, cfg.item[i], item_acc_test[i]))

    draw(item_acc_train_list, item_acc_test_list, item_loss_list, total_loss_list, model_name, cfg, epochs)


def vote(scores: list):
    """
    :param scores:      prediction of each gait cycle of the same person
    :return:
            choose the majority as the label
            if there is not only 1 mode, then use average as the final score.
    """
    new_scores = list()
    sub_cycles = len(scores)  # number of gait cycles
    items = len(scores[0])  # number of items
    for i in range(items):
        item_i_prediction = list()
        for j in range(sub_cycles):
            item_i_prediction.append(scores[j][i])
        # find the mode
        d = dict()
        for pred in item_i_prediction:
            d[pred] = d[pred] + 1 if pred in d else 1
        i_max = 0
        prediction = list()
        for key in d:
            # d[key] is the occurrence of score "key"
            if d[key] > i_max:
                i_max = d[key]
                prediction.clear()
                prediction.append(key)
            elif d[key] == i_max:
                prediction.append(key)
        new_scores.append(round(sum(prediction) / len(prediction)))

    return new_scores


def validation(model, dataset: Dataset, cfg: Config):
    """
    :param model:     same as the train function above
    :param dataset:
    :param cfg:
    :return:
    """
    with torch.no_grad():
        batches = len(dataset.test_data)
        item_acc_list = np.zeros(len(cfg.item))
        for batch in range(batches):
            test_data, test_labels, name, _ = dataset.load_data(train=False)
            # Note that, test data is a list of numpy arrays. We use majority vote to determine the final result
            predicted = list()
            for sub_cycle in test_data:
                sub_cycle = torch.tensor(sub_cycle, dtype=torch.float32).to(device, dtype=dtype)
                sub_out = model(sub_cycle)
                sub_predicted = list()
                for i in range(len(cfg.item)):
                    predict_i = torch.argmax(sub_out[i], dim=1).cpu().item()
                    sub_predicted.append(predict_i)
                predicted.append(sub_predicted)

            predicted = vote(predicted)
            for i in range(len(cfg.item)):
                if predicted[i] == test_labels[i]:
                    item_acc_list[i] += 1

        item_acc_list /= batches
        return item_acc_list
