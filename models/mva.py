
import torch
import torch.nn as nn


class MVA(nn.Module):

    def __init__(self, num_dense=8192, latent_dim=1024):
        super().__init__()

        self.latent_dim = latent_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        layers = []
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1):
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(1024, 8192*3))          
        self.decoder = nn.Sequential(*layers)



    def forward(self, xyz):
        xyz = xyz.transpose(1,2)
        B, N, _ = xyz.shape
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)

        pre = self.decoder(feature_global).view(-1,3,8192)

        return pre.contiguous()