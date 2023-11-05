"""
ST-GCN + Feature Refinement Head (Contrastive Learning)
"""

from baselines.ST_GCN_baseline import *
from baselines.FR_Head import *
import torch
import torch.nn as nn


class ST_FR_GCN(nn.Module):
    def __init__(self, in_channels, num_class: list, joints=23, edge_importance_weighting=True, max_hop=1,
                 classify_head="lstm", hidden=128, frames=120):
        """
        :param in_channels:
        :param num_class:
                            a list of items classes
        :param joints:
                            human body joints used
        :param edge_importance_weighting:
        :param max_hop:
        :param classify_head:
                            if dataset is padded, then rnn-like classification head can be used
                            otherwise, each gait cycle takes occupied different time
        :param hidden:      hidden size of rnn cell
        """
        super(ST_FR_GCN, self).__init__()
        self.head = classify_head
        self.graph = Graph(max_hop)
        # the adjacency matrix does not need to update
        adjacency = torch.tensor(self.graph.adjacency, dtype=torch.float32, requires_grad=False)
        # adjacency matrix isn't a model parameter and if we don't register, the model state_dict doesn't contain it
        # all in all, avoid updating while saving it in dict
        self.register_buffer("adjacency", adjacency)
        t_kernel = 9
        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.shape[1])

        self.l1 = ST_GCN_Block(t_kernel, 1, 1, in_channels, 64, residual=False)
        self.l2 = ST_GCN_Block(t_kernel, 1, 1, 64, 64, residual=True)
        self.l3 = ST_GCN_Block(t_kernel, 1, 2, 64, 128, residual=True)
        self.l4 = ST_GCN_Block(t_kernel, 1, 1, 128, 128, residual=True)

        self.fr1 = FR_Head(in_channels=64, frames=frames, joints=joints, classes=num_class[0])
        self.fr2 = FR_Head(in_channels=128, frames=frames // 2, joints=joints, classes=num_class[0])

        self.ei1 = nn.Parameter(torch.ones_like(adjacency))
        self.ei2 = nn.Parameter(torch.ones_like(adjacency))
        self.ei3 = nn.Parameter(torch.ones_like(adjacency))
        self.ei4 = nn.Parameter(torch.ones_like(adjacency))

        hidden_dim = 128

        self.fc = nn.Linear(in_features=hidden_dim, out_features=num_class[0])

    def forward(self, x, label=None, get_cl_loss=False):
        batch, channel, frames, joints = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch, channel * joints, frames)
        x = self.data_bn(x)
        x = x.view(batch, channel, joints, frames)
        x = x.permute(0, 1, 3, 2).contiguous()

        f1 = self.l1(x, self.adjacency * self.ei1)
        f2 = self.l2(f1, self.adjacency * self.ei2)
        f3 = self.l3(f2, self.adjacency * self.ei3)
        f4 = self.l4(f3, self.adjacency * self.ei4)

        feat_mid = f2.clone()
        feat_fin = f4.clone()

        out = F.avg_pool2d(f4, f4.size()[2:]).squeeze()
        # keep "batch = 1" shape
        if len(out.shape) == 1:
            out = torch.unsqueeze(out, dim=0)
        logits = self.fc(out)

        if get_cl_loss:
            cl_loss1 = self.fr1(feat_mid, label, logits)
            cl_loss2 = self.fr2(feat_fin, label, logits)

            cl_loss = cl_loss1 + cl_loss2

            return [logits], cl_loss
        else:
            return [logits]
