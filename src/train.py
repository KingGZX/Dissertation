import torch
import torch.optim as optim
import torch.nn as nn
from utils.plot import *
from loadata import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

train_acc_list = list()
train_loss_list = list()
test_acc_list = list()

names = ["train accuracy", "train loss", "test accuracy"]


def joint_train(net, dst, baseline: str, cfg: Config, epochs=20):
    net.to(device)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    batches = len(dst.train_data)

    item_train_acc_list_list = list([] for i in range(epochs))
    item_test_acc_list_list = list([] for i in range(epochs))

    for epoch in range(epochs):
        item_correct_list = list(0 for i in cfg.item)
        nloss = 0
        correct = 0
        dst.shuffle()
        for batch in range(batches):
            train_data, train_labels = dst.load_data()
            train_data = torch.tensor(train_data, dtype=torch.float32)
            for i in range(len(train_labels)):
                train_labels[i] = torch.tensor(train_labels[i])
                train_labels[i] = train_labels[i].to(device)
            train_data = train_data.to(device, dtype=dtype)

            if len(train_data.size()) == 3:
                train_data = train_data.unsqueeze(0)

            output = net(train_data)

            loss = torch.tensor(0)
            loss = loss.to(device, dtype=dtype)

            for i in range(len(train_labels)):
                item_loss = criterion(output[i], train_labels[i])
                loss += item_loss
                nloss += item_loss.item()

                """
                loss1 = criterion(output[0], train_labels[0])
                loss2 = criterion(output[1], train_labels[1])
                loss3 = criterion(output[2], train_labels[2])
                loss4 = criterion(output[3], train_labels[3])
                loss5 = criterion(output[4], train_labels[4])
                """

            # nloss += (loss1.item() + loss2.item() + loss3.item() + loss4.item() + loss5.item())

            # loss = loss1 + loss2 + loss3 + loss4 + loss5

            count = 0

            # if all the items are classified correctly, then "correct + 1"
            for i in range(len(train_labels)):
                if torch.argmax(output[i]) == train_labels[i]:
                    item_correct_list[i] += 1
                    count += 1

            if count == len(train_labels):
                correct += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_list.append(nloss / batches)
        train_acc_list.append(correct / batches)
        # accuracy of each item
        for item_correct in item_correct_list:
            item_train_acc_list_list[epoch].append(item_correct / batches)

        print("Epoch {}/{}:, average loss is {}".format(epoch + 1, epochs, nloss / batches))
        # print("Epoch {}/{}:, average accuracy is {}".format(epoch + 1, epochs, correct / batches))

        # for i in range(len(cfg.item)):
        # print("Epoch {}/{}:, average accuracy of item {} is {}".format(epoch + 1, epochs,
        # cfg.label_dict["item" + str(cfg.item[i])]
        # , item_train_acc_list_list[epoch][i]))

        item_acc_list = joint_validation(net, dst, cfg)
        item_test_acc_list_list[epoch] = item_acc_list

    # acc_loss_draw(epochs, [train_acc_list, train_loss_list, test_acc_list], names,
    # titlename=baseline)

    acc_loss_draw(epochs, [train_acc_list, train_loss_list, test_acc_list], names, title_name=baseline)
    items_acc_acc_draw(epochs, item_test_acc_list_list, cfg)
    items_acc_acc_draw(epochs, item_train_acc_list_list, cfg)

    # for Colab
    train_loss_list.clear()
    train_acc_list.clear()
    test_acc_list.clear()


def joint_validation(net, dst, cfg: Config):
    batches = len(dst.test_data)
    correct = 0

    item_correct_list = list(0 for i in cfg.item)
    item_acc_list = list()
    for batch in range(batches):
        test_data, test_labels = dst.load_data(train=False)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_data = test_data.to(device, dtype=dtype)
        for i in range(len(test_labels)):
            test_labels[i] = torch.tensor(test_labels[i])
            test_labels[i] = test_labels[i].to(device)
        if len(test_data.size()) == 3:
            test_data = test_data.unsqueeze(0)

        output = net(test_data)

        count = 0

        for i in range(len(test_labels)):
            if torch.argmax(output[i]) == test_labels[i]:
                item_correct_list[i] += 1
                count += 1

        if count == len(test_labels):
            correct += 1

    for item_correct in item_correct_list:
        item_acc_list.append(item_correct / batches)

    test_acc_list.append(correct / batches)
    return item_acc_list


def train(net, dst, baseline: str, epochs=20):
    """
    :param baseline:   baseline name, e.g. TCN, ST-GCN .......    use it as the title of the figure
    :param net:     ST-GCN Network
    :param dst:     Dataset Class
    :param epochs:  Hyperparameter
    :return:
    """
    net.to(device)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    batches = len(dst.train_data)

    for epoch in range(epochs):
        nloss = 0
        correct = 0
        dst.shuffle()
        for batch in range(batches):
            train_data, train_label = dst.load_data()
            train_data = torch.tensor(train_data, dtype=torch.float32)
            train_label = torch.tensor(train_label)
            train_data = train_data.to(device, dtype=dtype)
            train_label = train_label.to(device)
            if len(train_data.size()) == 3:
                train_data = train_data.unsqueeze(0)

            output = net(train_data)

            correct += (torch.argmax(output) == train_label).cpu()  # avoid 'cuda' error

            loss = criterion(output, train_label)
            nloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_list.append(nloss / batches)
        train_acc_list.append(correct / batches)

        print("Epoch {}/{}:, average loss is {}".format(epoch, epochs, nloss / batches))
        validation(net, dst)

    acc_loss_draw(epochs, [train_acc_list, train_loss_list, test_acc_list], names,
                  title_name=baseline)

    # for Colab
    train_loss_list.clear()
    train_acc_list.clear()
    test_acc_list.clear()


def validation(net, dst):
    """
    :param net:
    :param dst:
    :return:
        perform network validation
    """
    batches = len(dst.test_data)
    correct = 0
    for batch in range(batches):
        test_data, test_label = dst.load_data(train=False)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_label = torch.tensor(test_label)
        test_data = test_data.to(device, dtype=dtype)
        test_label = test_label.to(device)
        if len(test_data.size()) == 3:
            test_data = test_data.unsqueeze(0)

        output = net(test_data)
        correct += (torch.argmax(output) == test_label).cpu()  # avoid 'cuda' error

    acc = correct / batches
    test_acc_list.append(acc)
    print("Correct: {}/{}, Accuracy is {}".format(correct, batches, acc))
