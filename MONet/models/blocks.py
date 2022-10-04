import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-8

def to_prior_sigma(x, simgoid_bias=4.0, eps=1e-4):
    """
    This parameterisation bounds sigma of a learned prior to [eps, 1+eps].
    The default sigmoid_bias of 4.0 initialises sigma to be close to 1.0.
    The default eps prevents instability as sigma -> 0.
    """
    return torch.sigmoid(x + simgoid_bias) + eps

def to_var(x):
    return to_sigma(x)**2

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    out = torch.cat([grid, 1.0 - grid], dim=-1)
    return out


def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1

class ToSigma(nn.Module):
    def __init__(self):
        super(ToSigma, self).__init__()
    def forward(self, x):
        return to_sigma(x)

class ToVar(nn.Module):
    def __init__(self):
        super(ToVar, self).__init__()
    def forward(self, x):
        return to_var(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)

class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)
    def forward(self, x):
        b_sz = x.size(0)
        # Broadcast
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim, recompute_scale_factor=True)
        return self.pixel_coords(x)

class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim),
                                  torch.linspace(-1, 1, im_dim))
        self.g_1 = g_1.view((1, 1) + g_1.shape) # [1, 1, 72, 72]; range from -1, 1; uniform grid
        self.g_2 = g_2.view((1, 1) + g_2.shape)
    def forward(self, x):
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1).to(x.device) # [B*K, 1, 72, 72]
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1).to(x.device)
        return torch.cat((x, g_1, g_2), dim=1)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=True)

class INConvBlock(nn.Module):
    def __init__(self, nin, nout, stride=1, instance_norm=True, act=nn.ReLU()):
        super(INConvBlock, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not instance_norm)
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(nout, affine=True)
        else:
            self.instance_norm = None
        self.act = act
    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return self.act(x)

class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return inputs + emb_proj

class SlotEncoder(nn.Module):
    def __init__(
        self, 
        img_size=[64, 64], 
        in_channels=3,
        hidden_dims=[64, 64, 64, 64],
        kernel_size=5,
        ):

        super(SlotEncoder, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = hidden_dims[-1]
        self.kernel_size = kernel_size
        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_channels, self.img_size)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.LeakyReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
    
    def forward(self, x):
        batch_size, img_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)

        return encoder_out

class SlotDecoder(nn.Module):
    def __init__(
        self, 
        img_channels=3,
        img_size=[64,64],
        decoder_resolution=[4,4],
        hidden_dims=[64, 64, 64, 64],
        kernel_size=3,
        ):
        super(SlotDecoder, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[-1]
        self.kernel_size = kernel_size
        self.decoder_resolution = decoder_resolution
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, stride=2, padding=2, kernel_size=5, out_padding=1, dilation=1)
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
            )
        )

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.img_channels, self.out_features, self.decoder_resolution)

    
    def forward(self, slots):
        batch_size, num_slots, slot_dim = slots.shape

        slots = slots.view(batch_size * num_slots, slot_dim, 1, 1)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])


        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        out = out.view(batch_size, num_slots, self.img_channels + 1, self.img_size[0], self.img_size[1])
        recons = out[:, :, :self.img_channels, :, :]
        masks = out[:, :, -1:, :, :]
        
        return masks