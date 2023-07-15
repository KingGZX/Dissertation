import torch
from loadata import Config
import torch.nn as nn
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, t_kernel, time_steps, out_channels=None, num_classes=3, in_channels=3):
        super(TCN, self).__init__()
        joints = len(Config.nodes)
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

