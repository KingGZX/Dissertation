import torch
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    keep the sequence length (time frames) at each convolution
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        """
        In most cases, we wanna downsample the features and finally 
        make prediction
        """
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # 在整个Residual block中有非线性的激活. 这个容易忽略!
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x:
        :return: the expected input should have shape [batch, channles, seq_len(frames)]
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_class, frames=120, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.classify = nn.Linear(frames, num_class)

    def forward(self, x):
        x = torch.squeeze(self.network(x), dim=1)
        out = self.classify(x)
        return out


class TCN3D(nn.Module):
    def __init__(self, t_kernel, time_steps, out_channels=None, num_classes=3, in_channels=3):
        super(TCN3D, self).__init__()
        joints = 23
        if out_channels is None:
            out_channels = [64, 128, 128, 256, 256]
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(1, self.out_channels[0], kernel_size=(1, joints, in_channels))

        self.conv2 = nn.Conv1d(self.out_channels[0], self.out_channels[1], t_kernel)
        self.conv3 = nn.Conv1d(self.out_channels[1], self.out_channels[2], t_kernel)
        self.conv4 = nn.Conv1d(self.out_channels[2], self.out_channels[3], t_kernel)
        self.conv5 = nn.Conv1d(self.out_channels[3], self.out_channels[4], t_kernel)

        """
        since output size formula is W' = (W - kernel) + 1        // ignore stride = 1 && padding = 0
        W'' = W' - kernel + 1 = W - 2 * kernel + 2
        """
        output_frames = time_steps - 4 * (t_kernel - 1)
        self.output_features = output_frames * self.out_channels[4]

        self.fc = nn.Linear(self.output_features, self.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        :param x:
            for the unpadded dataset, the batch size = 1

            in most cases, the input shape is
            [1, channel, frames, joints]

            to perform 3D convolution, the first thing to do is to transform the input into
            [1, 1, frames, joints, channel]

            TCN is brute-force, but it's very similar to ST-GCN in some sense.
            it firstly uses Conv3D while keeping the time domain (just don't take graph adjacency into consideration)

            then stack Conv1D blocks tp perform temporal convolution
        :return:
        """
        x = torch.permute(x, [0, 2, 3, 1]).contiguous()
        x = torch.unsqueeze(x, dim=1)

        # 3d conv
        x = F.relu(self.conv1(x))
        x = torch.squeeze(x)

        # 1d conv
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = self.dropout(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.conv5(x)))
        x = x.view(-1, self.output_features)

        out = F.relu(self.fc(x))

        if out.shape[0] == 1:
            out = out.squeeze()

        return out
