import torch
import torch.nn as nn
import torch.nn.functional as F
from function import PointNetSetAbstraction, PointTransformerSALayer, PointNetFeaturePropagation


class FlowPredictor(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(FlowPredictor, self).__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=4096, radius=0.05, nsample=16,
            in_channel=in_channels + 3,
            mlp=[64, 64, 128],
            group_all=False
        )

        self.sa2 = PointTransformerSALayer(
            npoint=1024, radius=0.1, nsample=32,
            in_channels=128, out_channels=256
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )

        self.fp = PointNetFeaturePropagation(
            in_channel=1024 + in_channels,
            mlp=[1024, 1024]
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.LayerNorm(512)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.LayerNorm(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, out_channels)

    def forward(self, xyz):
        B, _, N = xyz.shape

        norm = xyz[:, 0:4, :]
        xyz_coords = xyz[:, 1:4, :]

        l1_xyz, l1_points = self.sa1(xyz_coords, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = self.fp(xyz_coords, l3_xyz, norm, l3_points)

        x = x.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = x.permute(0, 2, 1)

        return x


class PredictionLoss(nn.Module):
    def __init__(self):
        super(PredictionLoss, self).__init__()

    def forward(self, pred, target):
        e = pred - target
        mse = torch.mean(e ** 2)
        mae = torch.mean(torch.abs(e))
        return mse, mae


get_model = FlowPredictor
get_loss = PredictionLoss
