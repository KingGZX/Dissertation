import torch
import torch.optim as optim
import torch.nn as nn


def train(net, dst, epochs=20):
    """
    :param net:     ST-GCN Network
    :param dst:     Dataset Class
    :param epochs:  Hyperparameter
    :return:
    """
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    batches = len(dst.train_data)
    nloss = 0

    for epoch in range(epochs):
        nloss = 0
        for batch in range(batches):
            train_data, train_label = dst.load_data()
            train_data = torch.tensor(train_data, dtype=torch.float32)
            if len(train_data.size()) == 3:
                train_data = train_data.unsqueeze(0)
            train_label = torch.tensor(train_label)

            output = net(train_data)

            loss = criterion(output, train_label)
            nloss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {}/{}:, average loss is {}".format(epoch, epochs, nloss / batches))
        validation(net, dst)


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
        if len(test_data.size()) == 3:
            test_data = test_data.unsqueeze(0)

        output = net(test_data)

        correct += (torch.argmax(output) == test_label)
    print("Correct: {}/{}, Accuracy is {}".format(correct, batches, correct / batches))
