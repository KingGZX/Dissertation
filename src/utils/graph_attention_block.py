import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT_Block(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=16, dropout=0.3):
        super(GAT_Block, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(self.in_channels, self.hidden_dim)
        self.weight = nn.Linear(2 * self.hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x:
                original feature matrix, in shape [batch, channel, frames, nodes]
        :param mask:
                graph adjacency matrix, usually in shape [nodes, nodes]
                --------but for graph partition, have no idea now ---------
        :return:
        """
        x = torch.permute(x, [0, 2, 3, 1])         # [batch, frames, nodes, channel]
        batch, frames, nodes, channels = x.shape

        x = self.linear(x)

        x1 = torch.unsqueeze(x, dim=-2)
        x2 = torch.unsqueeze(x, dim=-3)

        x1 = x1.repeat(1, 1, 1, nodes, 1)
        x2 = x2.repeat(1, 1, nodes, 1, 1)

        x3 = torch.concat([x1, x2], dim=-1)

        x4 = self.weight(x3)
        x4 = torch.squeeze(x4, dim=-1)

        x4 = F.softmax(x4, dim=-1)
        x4 = torch.multiply(x4, mask)
        # apply adjacency matrix as the mask to each attention matrix.Element-wise multiplication to eliminate invalid..

        out = torch.einsum("bfnn, bfnj -> bfnj", x4, x)

        out = torch.permute(out, [0, 3, 1, 2])  # back to [batch, channel, frame, joints]
        return out


class GMAT_Block(nn.Module):
    """
    It's graph multi-head attention block
    """
    def __init__(self):
        super(GMAT_Block, self).__init__()
        pass


# code for debugging
"""
instance = GAT_Block()
x = torch.randn(2, 3, 200, 19)
out = instance(x)
print(out.shape)
"""
