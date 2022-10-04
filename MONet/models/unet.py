import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import INConvBlock, Flatten


class UNet(nn.Module):

    def __init__(self, num_blocks, filter_start=32, input_dim=4, output_dim=1):
        super(UNet, self).__init__()
        c = filter_start
        if num_blocks == 4:
            self.down = nn.ModuleList([
                INConvBlock(input_dim, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c)
            ])
        elif num_blocks == 5:
            self.down = nn.ModuleList([
                INConvBlock(input_dim, c),
                INConvBlock(c, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c)
            ])
        elif num_blocks == 6:
            self.down = nn.ModuleList([
                INConvBlock(input_dim, c),
                INConvBlock(c, c),
                INConvBlock(c, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c)
            ])
        elif num_blocks == 7:
            self.down = nn.ModuleList([
                INConvBlock(input_dim, c),
                INConvBlock(c, c),
                INConvBlock(c, c),
                INConvBlock(c, c),
                INConvBlock(c, 2*c),
                INConvBlock(2*c, 2*c),
                INConvBlock(2*c, 2*c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, 2*c),
                INConvBlock(4*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c),
                INConvBlock(2*c, c)
            ])
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(4*4*2*c, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4*4*2*c), nn.ReLU()
        )
        self.final_conv = nn.Conv2d(c, output_dim, 1)

    def forward(self, x): ## x shape: [batch_size, 4, img_size, img_size]
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        out_features = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
            x_down.append(act)
        out_features.extend(x_down)
        x_up = self.mlp(x_down[-1]).view(batch_size, -1, 4, 4)
        out_features.append(x_up)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest', recompute_scale_factor=True)
            out_features.append(x_up)
        ## x_up shape: [bs, 32, h, w]
        return self.final_conv(x_up), {}, out_features