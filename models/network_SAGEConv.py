import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch import Tensor


class Net(torch.nn.Module):
    def __init__(self, num_features: int = None, num_classes: int = None):
        super(Net, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv = SAGEConv(self.num_features,
                             self.num_classes,
                             aggr="max")  # max, mean, add ...)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x.x, x.edge_index)
        return F.log_softmax(x, dim=1)