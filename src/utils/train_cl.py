from train import *
from loadata import Config
from utils.log import get_logger
from loadata import Dataset
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def batch_train_CL(model, dataset: Dataset, epochs: int, model_name: str, cfg: Config, items: list, log=True,
                log_file=None, save=False):
    """
    :param model:        Network Instance
    :param dataset:      The class implemented in loadata.py
    :param epochs:       Hyperparameter
    :param model_name:   e.g. ST-GCN, Uniformer, GFN, for plotting result
    :param cfg:          config file
    :param items:        item needs to be trained by this model
    :param log_file:     Logging file name
    :param log:          Logging setting
    :param save:         whether to save the pth format of model
    :return:
    """
    # Hyperparameter setting
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=6e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = cfg.batch_size
    train_len = len(dataset.train_data)
    batches = math.ceil(train_len / batch_size)

    # move the model to GPU
    model.to(device)

    # create log file handler
    if log:
        import os
        if not os.path.exists("../log"):
            os.mkdir("../log")
        idx = len(os.listdir("../log")) + 1
        fp = "exp" + str(idx) + ".log" if log_file is None else log_file
        logger = get_logger(
            filename="../log/" + fp,
            verbosity=1
        )

    # for single item
    max_acc = 0

    item_loss_list, item_acc_train_list, total_loss_list, item_acc_test_list = list(), list(), list(), list()
    total_score_acc_list = list()
    for epoch in range(epochs):
        # shuffle gait cycles in the train set
        logger.info("Epoch: {}/{}".format(epoch + 1, epochs))
        dataset.shuffle()
        item_acc = np.zeros(len(items))
        item_loss = np.zeros(len(items))
        total_loss = 0
        for batch in range(batches):
            if batch_size > 1:
                train_data, train_label = dataset.load_batch_data_train(items=items)
            else:
                train_data, train_label, patient_name, gait_cycle = dataset.load_data(train=True, items=items)
            # to tensor
            train_data = torch.tensor(train_data, dtype=torch.float32).to(device, dtype=dtype)
            for i in range(len(items)):
                # each element of train_label is a list, it's corresponding labels of this batch data of items[i]
                train_label[i] = torch.tensor(train_label[i], dtype=torch.long).to(device)

            out, cl_loss = model(train_data, train_label[0], get_cl_loss=True)
            loss = torch.tensor(0).to(device, dtype=dtype)

            loss += cl_loss

            # this is especially for joint learning for which model could ouput the predict labels for more than 2 items
            for i in range(len(items)):
                i_loss = criterion(out[i], train_label[i])
                loss += i_loss
                correct_num = torch.sum(torch.argmax(out[i], dim=1) == train_label[i]).cpu()
                item_acc[i] += correct_num
                item_loss[i] += i_loss.item()

            total_loss += loss.item()
            total_loss += cl_loss.item()

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        item_acc = item_acc / train_len

        if save and item_acc[0] > max_acc:
            max_acc = item_acc[0]
            torch.save(model, "../best_model/item" + str(items[0]) + ".pth")

        item_loss = item_loss / batches
        total_loss = total_loss / batches
        item_acc_train_list.append(item_acc), item_loss_list.append(item_loss), total_loss_list.append(total_loss)
        item_acc_test, total_score_acc = validation(model, dataset, items, logger)
        total_score_acc_list.append(total_score_acc)
        item_acc_test_list.append(item_acc_test)
        print("Epoch:{}/{}, average loss is {}".format(epoch + 1, epochs, total_loss))
        for i in range(len(items)):
            print("Epoch:{}/{}, accuracy of item {} is {}".format(epoch + 1, epochs, items[i], item_acc_test[i]))

    draw(item_acc_train_list, item_acc_test_list, item_loss_list, total_loss_list, total_score_acc_list,
         model_name, cfg, items, epochs)