from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import numpy as np

from models.unet import UNet
import models.seq_att as seq_att
from models.component_vae import ComponentVAE
from utils.get_kl import get_kl


class MONet(nn.Module):

    def __init__(self, cfg, device):
        super(MONet, self).__init__()
        # Configuration
        self.cfg = cfg
        self.K_steps = cfg.K_steps
        self.prior_mode = cfg.prior_mode
        self.mckl = cfg.montecarlo_kl
        self.pixel_bound = cfg.pixel_bound
        # Sub-Modules
        # - Attention Network
        if not hasattr(cfg, 'filter_start'):
            cfg['filter_start'] = 32
        core = UNet(int(np.log2(cfg.img_size)-1), cfg.filter_start, input_dim=4, output_dim=1)
        self.att_process = seq_att.SimpleSBP(core)

        # - Component VAE
        self.comp_vae = ComponentVAE(nout=4, cfg=cfg, act=nn.ReLU())
        self.comp_vae.pixel_bound = False
        # Initialise pixel output standard deviations
        std = cfg.pixel_std2 * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = cfg.pixel_std1  # first step
        self.register_buffer('std', std)
        self.device = device


    @staticmethod
    def x_loss(x, log_m_k, x_r_k, std, pixel_wise=False):
        # 1.) Sum over steps for per pixel & channel (ppc) losses
        p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
        log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
        log_m_stack = torch.stack(log_m_k, dim=4)
        log_mx = log_m_stack + log_xr_stack
        err_ppc = -torch.log(log_mx.exp().sum(dim=4))
        # 2.) Sum accross channels and spatial dimensions
        if pixel_wise:
            return err_ppc
        else:
            return err_ppc.sum(dim=(1, 2, 3))

    def forward(self, x, labels=None):
        """
        Args:
            x (torch.Tensor): Input images [batch size, 3, dim, dim]
        """

        # --- Predict segmentation masks ---
        log_m_k, log_s_k, att_stats = self.att_process(x, self.K_steps-1)

        # log_m_k is list of len K: ([bs, 1, h, w]*K)
        # --- Reconstruct components ---
        x_m_r_k, comp_stats = self.comp_vae(x, log_m_k)
        # Split into appearances and mask prior
        x_r_k = [item[:, :3, :, :] for item in x_m_r_k] ## (K, [B, 3, dim, dim])
        m_r_logits_k = [item[:, 3:, :, :] for item in x_m_r_k] ## (K, [B, 1, dim, dim])
        # Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]

        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4) ## [B, 3, dim, dim, K]
        m_stack = torch.stack(log_m_k, dim=4).exp() ## [B, 1, dim, dim, K]
        m_stack_l2 = torch.square(m_stack).sum(dim=(1,2,3,4))
        recon = (m_stack * x_r_stack).sum(dim=4) ## size of product: [B, 3, dim, dim, 5], after summing up [B, 3, dim, dim]

        # --- Reconstruct masks ---
        log_m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, self.prior_mode, log=True) ## stack the list and normalize across K
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k] ## turn into list of length K

        # --- Loss terms ---
        losses = AttrDict()
        # -- Reconstruction loss
        losses['err'] = self.x_loss(x, log_m_k, x_r_k, self.std)
        # -- Attention mask KL
        losses['kl_m'] = self.kl_m_loss(log_m_k=log_m_k, log_m_r_k=log_m_r_k)
        losses['mask_l2'] = m_stack_l2
        # -- Component KL
        q_z_k = [Normal(m, s) for m, s in 
                zip(comp_stats.mu_k, comp_stats.sigma_k)]
        kl_l_k = get_kl(
            comp_stats.z_k, q_z_k, len(q_z_k)*[Normal(0, 1)], self.mckl)
        losses['kl_l_k'] = [kld.sum(1) for kld in kl_l_k]

        # Track quantities of interest
        stats = AttrDict(
            # recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            recon=recon, log_m_k=log_m_k, x_r_k=x_r_k,
            log_m_r_k=log_m_r_k,
            mx_r_k=[x*logm.exp() for x, logm in zip(x_r_k, log_m_k)])


        return recon, losses, stats, att_stats, comp_stats

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)

    def get_mask_recon_stack(self, m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_scope = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == self.K_steps - 1:
                    log_m_r_k.append(log_scope)
                else:
                    log_m = F.logsigmoid(logits)
                    log_neg_m = F.logsigmoid(-logits)
                    log_m_r_k.append(log_scope + log_m)
                    log_scope = log_scope +  log_neg_m
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError("No valid prior mode.")

    def kl_m_loss(self, log_m_k, log_m_r_k):
        batch_size = log_m_k[0].size(0)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        # Lower bound to 1e-5 to avoid infinities
        m_stack = torch.max(m_stack, torch.tensor(1e-5).to(m_stack.device))
        m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5).to(m_r_stack.device))
        q_m = Categorical(m_stack.view(-1, self.K_steps))
        p_m = Categorical(m_r_stack.view(-1, self.K_steps))
        kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
        return kl_m_ppc.sum(dim=1)

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        # Sample latents
        z_batched = Normal(0, 1).sample((batch_size*K_steps, self.comp_vae.ldim)).to(self.device)
        # Pass latent through decoder
        x_hat_batched = self.comp_vae.decode(z_batched)
        # Split into appearances and masks
        x_r_batched = x_hat_batched[:, :3, :, :]
        m_r_logids_batched = x_hat_batched[:, 3:, :, :]
        # Apply pixel bound to appearances
        if self.pixel_bound:
            x_r_batched = torch.sigmoid(x_r_batched)
        # Chunk into K steps
        x_r_k = torch.chunk(x_r_batched, K_steps, dim=0)
        m_r_logits_k = torch.chunk(m_r_logids_batched, K_steps, dim=0)
        # Normalise masks
        m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, self.prior_mode, log=False)
        # Apply masking and sum to get generated image
        x_r_stack = torch.stack(x_r_k, dim=4)
        gen_image = (m_r_stack * x_r_stack).sum(dim=4)
        # Tracking
        log_m_r_k = [item.squeeze(dim=4) for item in
                     torch.split(m_r_stack.log(), 1, dim=4)]
        stats = AttrDict(gen_image=gen_image, x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)])
        return gen_image, stats
