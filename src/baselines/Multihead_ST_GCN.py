import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_attention_block import GAT_Block
from graph import Graph


class TemporalUnit(nn.Module):
    def __init__(self, t_kernel, stride, in_channel):
        super(TemporalUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.active = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=(t_kernel, 1),
            stride=(stride, 1),
            padding=(t_kernel//2, 0)    # pad the time domain
        )
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.dropout = nn.Dropout(0.5, inplace=True)

    def forward(self, x):
        """
        :param x:  x is actually the features after normal GCN
                   it's in shape [batch, channel, frames, joints]

                   therefore, the main job of this unit is to perform convolution
                   on time domain.
        :return:
        """
        b1 = self.active(self.bn1(x))
        b2 = self.conv(b1)
        out = self.dropout(self.bn2(b2))
        return out


class GCNUnit(nn.Module):
    def __init__(self, out_channel, kernel_size, in_channel=3):
        """
        :param out_channel:
                for each adjacent matrix, we have corresponding feature maps with out_channels channel
        :param kernel_size:
                actually it's the num of Adjacent Matrices.
                The original paper use graph partition technique and have various adjacent matrices
        :param in_channel:
                default is 3, because we only have 3D position information at the very first time
        """
        super(GCNUnit, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel * kernel_size,
            kernel_size=(1, 1),
            stride=1
        )

    def forward(self, x, adjacent):
        """
        :param x:
                input features in shape [batch, channel, frames, joints]
        :param adjacent:
                adjacent matrices
        :return:
        """
        x1 = self.conv(x)
        b, c, w, h = x1.shape
        x1 = x1.view(b, self.kernel_size, c // self.kernel_size, w, h)

        out = torch.einsum("bkcfj, kjw -> bcfw", (x1, adjacent))
        return out


class ST_GCN_Block(nn.Module):
    def __init__(self, t_kernel, s_kernel, stride, in_channel, out_channel, residual=True):
        """
        :param t_kernel:        temporal kernel used in temporal convolution unit
        :param s_kernel:        spatial kernel which is same as num of adjacent matrices
        :param stride:
        :param in_channel:

        an ST-GCN block is consisted of a TemporalUnit, GCNUnit and a residual link
        """
        super(ST_GCN_Block, self).__init__()
        self.gcn = GCNUnit(out_channel, s_kernel, in_channel)
        self.tcn = TemporalUnit(t_kernel, stride, out_channel)
        if not residual:
            self.residual = lambda x: 0
        elif out_channel == in_channel and stride == 1:
            # we will automatically do padding in tcn to fit the temporal kernel
            self.residual = nn.Identity()
        else:
            # for tcn, the time axis size formula is
            # (frames + 2 * (t_kernel // 2) - t_kernel) / stride + 1
            # we force the t_kernel to be an odd number, then it can be simplified to : (frames - 1) / stride + 1
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    1,
                    (stride, 1),            # (frames - 1) / stride + 1
                ),
                nn.BatchNorm2d(out_channel)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adjacent):
        x1 = self.gcn(x, adjacent)
        x2 = self.residual(x)
        out = self.relu(self.tcn(x1) + x2)
        return out


class step_length(nn.Module):
    def __init__(self, in_channels, classes):
        super(step_length, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class weight_shift(nn.Module):
    def __init__(self, in_channels, classes):
        super(weight_shift, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class stance_width(nn.Module):
    def __init__(self, in_channels, classes):
        super(stance_width, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class hip_hiking(nn.Module):
    def __init__(self, in_channels, classes):
        super(hip_hiking, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class pelvic_rotation(nn.Module):
    def __init__(self, in_channels, classes):
        super(pelvic_rotation, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class stance_time(nn.Module):
    def __init__(self, in_channels, classes):
        super(stance_time, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class guard(nn.Module):
    def __init__(self, in_channels, classes):
        super(guard, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class hip_extension(nn.Module):
    def __init__(self, in_channels, classes):
        super(hip_extension, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class external_rotation(nn.Module):
    def __init__(self, in_channels, classes):
        super(external_rotation, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class circumduction(nn.Module):
    def __init__(self, in_channels, classes):
        super(circumduction, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class knee_flexion(nn.Module):
    def __init__(self, in_channels, classes):
        super(knee_flexion, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class toe_clearance(nn.Module):
    def __init__(self, in_channels, classes):
        super(toe_clearance, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, features):
        """

        :param features:
                        meaningful high dimension features extracted by ST-GCN blocks
                        assume it's in shape [batch, in_channels]
        :return:
                        predicted class
        """
        return self.mlp(features)


class MultiHead_ST_GCN(nn.Module):
    def __init__(self, in_channels, edge_importance_weighting=True, max_hop=1):
        """
        stack many st_gcn blocks together
        """
        super(MultiHead_ST_GCN, self).__init__()
        self.graph = Graph(max_hop)
        # the adjacency matrix does not need to update
        adjacency = torch.tensor(self.graph.adjacency, dtype=torch.float32, requires_grad=False)
        # adjacency matrix isn't a model parameter and if we don't register, the model state_dict doesn't contain it
        # all in all, avoid updating while saving it in dict
        self.register_buffer("adjacency", adjacency)
        t_kernel = 9
        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.shape[1])
        self.st_gcn = nn.ModuleList((
            # spatial_kernel_size = 1 means we only have one adjacency matrix
            ST_GCN_Block(t_kernel, 1, 1, in_channels, 64, residual=False),

            GAT_Block(in_channels=64, hidden_dim=64),
            # ST_GCN_Block(t_kernel, 1, 1, 64, 64),
            # ST_GCN_Block(t_kernel, 1, 1, 64, 64),
            # ST_GCN_Block(t_kernel, 1, 1, 64, 64),
            ST_GCN_Block(t_kernel, 1, 2, 64, 128),

            GAT_Block(in_channels=128, hidden_dim=128),
            # ST_GCN_Block(t_kernel, 1, 1, 128, 128),
            # ST_GCN_Block(t_kernel, 1, 1, 128, 128),
            ST_GCN_Block(t_kernel, 1, 2, 128, 256),

            ST_GCN_Block(t_kernel, 1, 1, 256, 256),
            # ST_GCN_Block(t_kernel, 1, 1, 256, 256),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(adjacency))
                for i in self.st_gcn
            ])

        # self.fcn = nn.Conv2d(256, num_class, 1)
        self.item2 = stance_time(256, classes=3)
        self.item3 = step_length(256, classes=3)
        self.item4 = weight_shift(256, classes=3)
        self.item5 = stance_width(256, classes=3)
        self.item6 = guard(256, classes=3)
        self.item7 = hip_extension(256, classes=3)
        self.item8 = external_rotation(256, classes=3)
        self.item9 = circumduction(256, classes=3)
        self.item10 = hip_hiking(256, classes=3)
        self.item11 = knee_flexion(256, classes=4)
        self.item12 = toe_clearance(256, classes=3)
        self.item13 = pelvic_rotation(256, classes=3)

    def forward(self, x):
        """
        :param x:     input batch gait cycles
        :return:
                      encoded features
        """
        batch, channel, frames, joints = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch, channel * joints, frames)
        x = self.data_bn(x)
        x = x.view(batch, channel, joints, frames)
        x = x.permute(0, 1, 3, 2).contiguous()

        # forward
        for gcn, importance in zip(self.st_gcn, self.edge_importance):
            x = gcn(x, self.adjacency * importance)

        # global pooling
        # average each feature map as the feature.   will be in shape (batch, channel, 1, 1)
        x = F.avg_pool2d(x, x.size()[2:])

        # x = self.fcn(x)
        features = x.squeeze()

        # perform joint learning
        out1 = self.item2(features)
        out2 = self.item3(features)
        out3 = self.item4(features)
        out4 = self.item5(features)
        out5 = self.item6(features)
        out6 = self.item7(features)
        out7 = self.item8(features)
        out8 = self.item9(features)
        out9 = self.item10(features)
        out10 = self.item11(features)
        out11 = self.item12(features)
        out12 = self.item13(features)

        out = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12]

        return out


# code for debugging
# x = torch.randn(3, 3, 390, 19)
# net = ST_GCN(x.shape[1], 10)
# print(net(x).shape)
