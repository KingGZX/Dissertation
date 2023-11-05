import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT_Block(nn.Module):
    """
    Graph Attention Block, including multi-head attention
    """
    def __init__(self, n_heads=1, in_channels=3, hidden_dim=16, dropout=0.3):
        super(GAT_Block, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim // n_heads
        self.heads = n_heads
        self.linear1 = nn.Linear(self.in_channels, self.hidden_dim * n_heads)
        self.linear2 = nn.Linear(2 * self.hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.LRelu = nn.LeakyReLU()

    def forward(self, x, mask):
        """
        :param x:
                original feature matrix, in shape [batch, channel, frames, nodes]
        :param mask:
                graph adjacency matrix, usually in shape [nodes, nodes]
                --------but for graph partition, have no idea now ---------
        :return:
        """
        assert self.hidden_dim % self.heads == 0, "head is not divisible by hidden dim"
        x = torch.permute(x, [0, 2, 3, 1])  # [batch, frames, nodes, channel]
        batch, frames, nodes, channels = x.shape

        # [batch, frames, nodes, hidden_dim * heads]
        x = self.linear1(x).view(batch, frames, nodes, self.heads, self.hidden_dim)
        # reshape
        x = torch.permute(x, [0, 3, 1, 2, 4]).contiguous()

        x1 = torch.unsqueeze(x, dim=-2)
        x2 = torch.unsqueeze(x, dim=-3)

        x1 = x1.repeat(1, 1, 1, 1, nodes, 1)
        x2 = x2.repeat(1, 1, 1, nodes, 1, 1)

        x3 = torch.concat([x1, x2], dim=-1)

        x4 = self.LRelu(self.linear2(x3))
        # [batch, heads, frames, nodes, nodes], the feature map stands for the contribution of Node i to Node j
        x4 = torch.squeeze(x4, dim=-1)

        # apply adjacency matrix as the mask to each attention matrix.
        # Element-wise multiplication to eliminate invalid contribution of neighbor node
        x4 = torch.multiply(x4, mask)

        x4 = F.softmax(x4, dim=-1)

        x4 = self.dropout(x4)

        out = torch.einsum("bhfnn, bhfnj -> bhfnj", x4, x)      # [batch, head, frame, nodes, hidden_dim]

        out = torch.permute(out, [0, 2, 3, 1, 4]).contiguous()
        out = out.view(batch, frames, nodes, -1)                # [batch, frames, nodes, hidden_dim * heads]

        out = torch.permute(out, [0, 3, 1, 2])  # back to [batch, channel, frame, joints]
        return out


# code for debugging
"""
instance = GAT_Block(n_heads=3)
x = torch.randn(2, 3, 200, 19)
mask = torch.ones((19, 19))
out = instance(x, mask)
print(out.shape)
"""
