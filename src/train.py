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


def train(net, dst, item, epochs=20, baseline="ST-GCN"):
    """
    :param net:     ST-GCN Network
    :param dst:     Dataset Class
    :param epochs:  Hyperparameter
    :param item:    wisconsin scale item
    :return:
    """
    net.to(device)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    batches = len(dst.train_data)
    nloss = 0

    for epoch in range(epochs):
        nloss = 0
        correct = 0
        dst.shuffle()
        for batch in range(batches):
            train_data, train_label = dst.load_data(item)
            train_data = torch.tensor(train_data, dtype=torch.float32)
            train_label = torch.tensor(train_label)
            train_data = train_data.to(device, dtype=dtype)
            train_label = train_label.to(device)
            if len(train_data.size()) == 3:
                train_data = train_data.unsqueeze(0)

            output = net(train_data)

            correct += (torch.argmax(output) == train_label).cpu()             # avoid 'cuda' error

            loss = criterion(output, train_label)
            nloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_list.append(nloss / batches)
        train_acc_list.append(correct / batches)

        print("Epoch {}/{}:, average loss is {}".format(epoch, epochs, nloss / batches))
        validation(net, dst, item)

    acc_loss_draw(epochs, [train_acc_list, train_loss_list, test_acc_list], names,
                  titlename=baseline + " on " + Config.label_dict["item" + str(item)])

    # for Colab
    train_loss_list.clear()
    train_acc_list.clear()
    test_acc_list.clear()


def validation(net, dst, item):
    """
    :param net:
    :param dst:
    :param item:
    :return:
        perform network validation
    """
    batches = len(dst.test_data)
    correct = 0
    for batch in range(batches):
        test_data, test_label = dst.load_data(item, train=False)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_label = torch.tensor(test_label)
        test_data = test_data.to(device, dtype=dtype)
        test_label = test_label.to(device)
        if len(test_data.size()) == 3:
            test_data = test_data.unsqueeze(0)

        output = net(test_data)
        correct += (torch.argmax(output) == test_label).cpu()                      # avoid 'cuda' error

    acc = correct / batches
    test_acc_list.append(acc)
    print("Correct: {}/{}, Accuracy is {}".format(correct, batches, acc))
