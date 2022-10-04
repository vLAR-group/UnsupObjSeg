from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq
from torch.distributions.normal import Normal

import models.blocks as B

class MONetCompEncoder(nn.Module):

    def __init__(self, cfg, act):
        super(MONetCompEncoder, self).__init__()
        nin = cfg.input_channels if hasattr(cfg, 'input_channels') else 3
        c = cfg.comp_enc_channels
        self.ldim = cfg.comp_ldim
        nin_mlp = 2*c * (cfg.img_size//16)**2
        nhid_mlp = max(256, 2*self.ldim)
        # self.module = Seq(nn.Conv2d(nin+1, c, 3, 2, 1), act,
        #                   nn.Conv2d(c, c, 3, 2, 1), act,
        #                   nn.Conv2d(c, 2*c, 3, 2, 1), act,
        #                   nn.Conv2d(2*c, 2*c, 3, 2, 1), act,
        #                   B.Flatten(),
        #                   nn.Linear(nin_mlp, nhid_mlp), act,
        #                   nn.Linear(nhid_mlp, 2*self.ldim))
        # self.module = Seq(nn.Conv2d(nin+1, c, 7, 4, 2), act,
        #                   nn.Conv2d(c, c, 3, 2, 1), act,
        #                   nn.Conv2d(c, 2*c, 3, 2, 1), act,
        #                   B.Flatten(),
        #                   nn.Linear(nin_mlp, nhid_mlp), act,
        #                   nn.Linear(nhid_mlp, 2*self.ldim)
        #                 )
        # self.module = Seq(nn.Conv2d(nin+1, c, 2, 2, 0), act,
        #                   nn.Conv2d(c, c, 3, 2, 1), act,
        #                   nn.Conv2d(c, 2*c, 3, 2, 1), act,
        #                   nn.Conv2d(2*c, 2*c, 3, 2, 1), act,
        #                   B.Flatten(),
        #                   nn.Linear(nin_mlp, nhid_mlp), act,
        #                   nn.Linear(nhid_mlp, 2*self.ldim)
        #                 )
        # self.module = Seq(nn.Conv2d(nin+1, c, 11, 4, 2), act,
        #                   nn.Conv2d(c, c, 3, 2, 1), act,
        #                   nn.Conv2d(c, 2*c, 3, 2, 1), act,
        #                   B.Flatten(),
        #                   nn.Linear(nin_mlp, nhid_mlp), act,
        #                   nn.Linear(nhid_mlp, 2*self.ldim)
        #                 )
        # self.module = Seq(nn.Conv2d(nin+1, c, 1, 1, 0), act,
        #                   nn.Conv2d(c, c, 2, 2, 0), act,
        #                   nn.Conv2d(c, c, 2, 2, 0), act,
        #                   nn.Conv2d(c, 2*c, 2, 2, 0), act,
        #                   nn.Conv2d(2*c, 2*c, 2, 2, 0), act,
        #                   B.Flatten(),
        #                   nn.Linear(nin_mlp, nhid_mlp), act,
        #                   nn.Linear(nhid_mlp, 2*self.ldim)
        #                 )
        self.module = Seq(nn.Conv2d(nin+1, c, 1, 1, 0), act,
                          nn.Conv2d(c, c, 1, 1, 0), act,
                          nn.Conv2d(c, c, 1, 1, 0), act,
                          nn.Conv2d(c, 2*c, 2, 2, 0), act,
                          nn.Conv2d(2*c, 2*c, 8, 8, 0), act,
                          B.Flatten(),
                          nn.Linear(nin_mlp, nhid_mlp), act,
                          nn.Linear(nhid_mlp, 2*self.ldim)
                        )
    def forward(self, x):
        out = self.module(x)
        return out

class BroadcastDecoder(nn.Module):

    def __init__(self, in_chnls, out_chnls, h_chnls, num_layers, img_dim, act):
        super(BroadcastDecoder, self).__init__()
        broad_dim = img_dim + 2*num_layers
        mods = [B.BroadcastLayer(broad_dim),
                nn.Conv2d(in_chnls+2, h_chnls, 3),
                act]
        for _ in range(num_layers - 1):
            mods.extend([nn.Conv2d(h_chnls, h_chnls, 3), act])
        mods.append(nn.Conv2d(h_chnls, out_chnls, 1))
        self.seq = nn.Sequential(*mods)

    def forward(self, x):
        return self.seq(x)

class ComponentVAE(nn.Module):

    def __init__(self, nout, cfg, act):
        super(ComponentVAE, self).__init__()
        self.ldim = cfg.comp_ldim  # paper uses 16
        self.texture_ldim = cfg.comp_texture_ldim
        self.shape_ldim = cfg.comp_shape_ldim
        assert self.ldim == self.texture_ldim + self.shape_ldim
        self.use_seperate_latent_code = cfg.seperate_latent_code
        self.montecarlo = cfg.montecarlo_kl
        self.pixel_bound = cfg.pixel_bound
        # Sub-Modules
        self.encoder_module = MONetCompEncoder(cfg=cfg, act=act)
        self.texture_decoder_module = BroadcastDecoder(
            in_chnls=self.texture_ldim,
            out_chnls=3,
            h_chnls=cfg.comp_dec_channels,
            num_layers=cfg.comp_dec_layers,
            img_dim=cfg.img_size,
            act=act
        )
        self.shape_decoder_module = BroadcastDecoder(
            in_chnls=self.shape_ldim,
            out_chnls=1,
            h_chnls=cfg.comp_dec_channels,
            num_layers=cfg.comp_dec_layers,
            img_dim=cfg.img_size,
            act=act
        )
        self.decoder_module = BroadcastDecoder(
            in_chnls=self.ldim,
            out_chnls=nout,
            h_chnls=cfg.comp_dec_channels,
            num_layers=cfg.comp_dec_layers,
            img_dim=cfg.img_size,
            act=act
        )
        

    def forward(self, x, log_mask):
        """
        Args:
            x (torch.Tensor): Input to reconstruct [batch size, 3, dim, dim]
            log_mask (torch.Tensor or list of torch.Tensors):
                Mask to reconstruct [batch size, 1, dim, dim]
        """
        # -- Check if inputs are lists
        K = 1
        b_sz = x.size(0)
        if isinstance(log_mask, list) or isinstance(log_mask, tuple):
            K = len(log_mask)
            # Repeat x along batch dimension
            x = x.repeat(K, 1, 1, 1)
            # Concat log_m_k along batch dimension
            log_mask = torch.cat(log_mask, dim=0)
        ## x shape: [bs*K, 3, h, w]; log mask shape: [bs*K, 1, h, w]
        # -- Encode
        x = torch.cat((log_mask, x), dim=1)  # Concat along feature dimension
        mu, sigma = self.encode(x) # mu: [bs*K, 16], sigma: [20, 16]
        
        if self.use_seperate_latent_code:
            mu_texture = mu[:, :self.texture_ldim]
            mu_shape = mu[:, self.texture_ldim:]
            sigma_texture = sigma[:, :self.texture_ldim]
            sigma_shape = sigma[:, self.texture_ldim:]
            q_z_texture = Normal(mu_texture, sigma_texture)
            q_z_shape = Normal(mu_shape, sigma_shape)
            z_texture = q_z_texture.rsample()
            z_shape = q_z_shape.rsample()
            texture_r = self.decode_texture(z_texture)
            shape_r = self.decode_shape(z_shape)
            x_r = torch.cat([texture_r, shape_r], dim=1)
        else:
            # -- Sample latents
            q_z = Normal(mu, sigma)
            # z - [batch_size * K, l_dim] with first axis: b0,k0 -> b0,k1 -> ...
            z = q_z.rsample()

            # -- Decode
            # x_r, m_r_logits = self.decode(z)
            x_r = self.decode(z) # x_r shape: [bs*K, 4, h, w]

        # -- Track quantities of interest and return
        x_r_k = torch.chunk(x_r, K, dim=0) # is a list of len K: ([bs, 4, h, w]*K)
        if self.use_seperate_latent_code:
            z_texure_k = torch.chunk(z_texture, K, dim=0)
            z_shape_k = torch.chunk(z_shape, K, dim=0)
            mu_texture_k = torch.chunk(mu_texture, K, dim=0)
            mu_shape_k = torch.chunk(mu_shape, K, dim=0)
            sigma_texture_k = torch.chunk(sigma_texture, K, dim=0)
            sigma_shape_k = torch.chunk(sigma_shape, K, dim=0)
            stats = AttrDict(mu_texture_k=mu_texture_k, sigma_texture_k=sigma_texture_k, z_texure_k=z_texure_k,
                            mu_shape_k=mu_shape_k, sigma_shape_k=sigma_shape_k, z_shape_k=z_shape_k)
        else:
            z_k = torch.chunk(z, K, dim=0)
            mu_k = torch.chunk(mu, K, dim=0)
            sigma_k = torch.chunk(sigma, K, dim=0)
            stats = AttrDict(mu_k=mu_k, sigma_k=sigma_k, z_k=z_k)
        return x_r_k, stats

    def encode(self, x):
        x = self.encoder_module(x)
        mu, sigma_ps = torch.chunk(x, 2, dim=1)
        sigma = B.to_sigma(sigma_ps)
        return mu, sigma

    def decode(self, z):
        x_hat = self.decoder_module(z)
        if self.pixel_bound:
            x_hat = torch.sigmoid(x_hat)
        return x_hat
    
    def decode_texture(self, z_texture):
        x_hat = self.texture_decoder_module(z_texture)
        if self.pixel_bound:
            x_hat = torch.sigmoid(x_hat)
        return x_hat
    
    def decode_shape(self, z_shape):
        x_hat = self.shape_decoder_module(z_shape)
        if self.pixel_bound:
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def sample(self, batch_size=1, steps=1):
        raise NotImplementedError