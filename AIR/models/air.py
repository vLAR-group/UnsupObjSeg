from collections import namedtuple

import torch
import os
import torch.nn.functional as F
from torch import nn
import re
import numpy as np
# from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import LSTMCell

from utils.misc import nograd_param, nan_checker
from utils.misc import BernoulliWrapper as Bernoulli
from utils.spatial_transform import SpatialTransformer, batch_cut_out_objects

State = namedtuple(
    'State',
    ['h', 'c', 'bl_h', 'bl_c', 'z_pres', 'z_where', 'z_what', 'z_what_loc'])


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_step = 0

    def increment_global_step(self):
        self.global_step += 1

    def get_device(self):
        return next(self.parameters()).device

    def save_checkpoint(self, ckpt_folder, epoch):
        path = os.path.join(ckpt_folder, "model_{}.pt".format(epoch))
        torch.save(self.state_dict(), path)

    def load(self, ckpt_folder, device=None, epoch=None):
        if epoch is None:
            filenames = list(filter(lambda n: "model_" in n, os.listdir(ckpt_folder)))
            regex = re.compile(r'\d+')
            numbers = [int(regex.search(n).group(0)) for n in filenames]
            ckpt_name = filenames[np.argmax(numbers)]   # get latest checkpoint
            epoch = max(numbers)
        else:
            ckpt_name = "model_{}.pt".format(epoch)
        print("Loading model checkpoint at epoch {}...".format(epoch))
        path = os.path.join(ckpt_folder, ckpt_name)
        self.load_state_dict(torch.load(path, map_location=device))
        # self.global_step = step
        print("Loaded.")
        
class Predictor(nn.Module):
    """
    Infer presence and location from LSTM hidden state
    """
    def __init__(self, lstm_hidden_dim):
        nn.Module.__init__(self)
        self.seq = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 7),
        )
        
    def forward(self, h):
        z = self.seq(h)
        z_pres_p = torch.sigmoid(z[:, :1])
        z_where_loc = z[:, 1:4]
        z_where_scale = F.softplus(z[:, 4:])
        return z_pres_p, z_where_loc, z_where_scale

    
class AppearanceEncoder(nn.Module):
    """
    Infer object appearance latent z_what given an image crop around the object
    """
    def __init__(self, object_size, color_channels, encoder_hidden_dim, z_what_dim):
        super().__init__()
        object_numel = color_channels * (object_size ** 2)
        self.net = nn.Sequential(
            nn.Linear(object_numel, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, z_what_dim * 2)
        )

    def forward(self, crop):
        """
        :param crop: (B, C, H, W)
        :return: z_what_loc, z_what_scale
        """
        bs = crop.size(0)
        crop_flat = crop.view(bs, -1)
        x = self.net(crop_flat)
        z_what_loc, z_what_scale = x.chunk(2, dim=1)
        z_what_scale = F.softplus(z_what_scale)
        
        return z_what_loc, z_what_scale
    
class AppearanceDecoder(nn.Module):
    """
    Generate pixel representation of an object given its latent code z_what
    """
    def __init__(self, z_what_dim, decoder_hidden_dim,
                 object_size, color_channels, bias=-2.0):
        super().__init__()
        object_numel = color_channels * (object_size ** 2)
        self.net = nn.Sequential(
            nn.Linear(z_what_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, object_numel),
        )
        self.sz = object_size
        self.ch = color_channels
        self.bias = bias

    def forward(self, z_what):
        x = self.net(z_what)
        x = x.view(-1, self.ch, self.sz, self.sz)
        x = torch.sigmoid(x + self.bias)
        return x

class AIR(BaseModel):
    """
    AIR model. Default settings are from the pyro tutorial. With those settings
    we can reproduce results from the original paper (although about 1/10 times
    it doesn't converge to a good solution).
    """

    z_where_dim = 3
    z_pres_dim = 1
    
    def __init__(self,
                 img_size,
                 object_size,
                 max_steps,
                 color_channels,
                 likelihood=None,
                 z_what_dim=50,
                 lstm_hidden_dim=256,
                 baseline_hidden_dim=256,
                 encoder_hidden_dim=200,
                 decoder_hidden_dim=200,
                 scale_prior_mean=3.0,
                 scale_prior_std=0.2,
                 pos_prior_mean=0.0,
                 pos_prior_std=1.0,
                 z_pres_prob_prior=0.01,
                 ):
        super().__init__()

        #### Settings

        self.max_steps = max_steps

        self.img_size = img_size
        self.object_size = object_size
        self.color_channels = color_channels
        self.z_what_dim = z_what_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.baseline_hidden_dim = baseline_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.z_pres_prob_prior = nograd_param(z_pres_prob_prior)
        self.z_where_loc_prior = nograd_param(
            [scale_prior_mean, pos_prior_mean, pos_prior_mean])
        self.z_where_scale_prior = nograd_param(
            [scale_prior_std, pos_prior_std, pos_prior_std])
        self.z_what_loc_prior = nograd_param(0.0)
        self.z_what_scale_prior = nograd_param(1.0)
        ####

        self.img_numel = color_channels * (img_size ** 2)

        lstm_input_size = (self.img_numel + self.z_what_dim
                           + self.z_where_dim + self.z_pres_dim)
        self.lstm = LSTMCell(lstm_input_size, self.lstm_hidden_dim)

        # Infer presence and location from LSTM hidden state
        self.predictor = Predictor(self.lstm_hidden_dim)

        # Infer z_what given an image crop around the object
        self.encoder = AppearanceEncoder(object_size, color_channels,
                                         encoder_hidden_dim, z_what_dim)

        # Generate pixel representation of an object given its z_what
        self.decoder = AppearanceDecoder(z_what_dim, decoder_hidden_dim,
                                         object_size, color_channels)

        # Spatial transformer (does both forward and inverse)
        self.spatial_transf = SpatialTransformer(
            (self.object_size, self.object_size),
            (self.img_size, self.img_size))

        # Baseline LSTM
        self.bl_lstm = LSTMCell(lstm_input_size, self.baseline_hidden_dim)

        # Baseline regressor
        self.bl_regressor = nn.Sequential(
            nn.Linear(self.baseline_hidden_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

        # Prior distributions
        self.pres_prior = Bernoulli(probs=self.z_pres_prob_prior)
        self.where_prior = Normal(loc=self.z_where_loc_prior,
                                  scale=self.z_where_scale_prior)
        self.what_prior = Normal(loc=self.z_what_loc_prior,
                                 scale=self.z_what_scale_prior)

        # Data likelihood
        self.likelihood = likelihood

    @staticmethod
    def _module_list_to_params(modules):
        params = []
        for module in modules:
            params.extend(module.parameters())
        return params

    def air_params(self):
        air_modules = [self.predictor, self.lstm, self.encoder, self.decoder]
        return self._module_list_to_params(air_modules) + [self.z_pres_prob_prior]

    def baseline_params(self):
        baseline_modules = [self.bl_regressor, self.bl_lstm]
        return self._module_list_to_params(baseline_modules)

    def get_output_dist(self, mean):
        if self.likelihood == 'original':
            std = torch.tensor(0.3).to(self.get_device())
            dist = Normal(mean, std.expand_as(mean))
        elif self.likelihood == 'bernoulli':
            dist = Bernoulli(probs=mean)
        else:
            msg = "Unrecognized likelihood '{}'".format(self.likelihood)
            raise RuntimeError(msg)
        return dist
        
    def forward(self, x):
        bs = x.size(0)

        # Init model state
        state = State(
            h=torch.zeros(bs, self.lstm_hidden_dim, device=x.device),
            c=torch.zeros(bs, self.lstm_hidden_dim, device=x.device),
            bl_h=torch.zeros(bs, self.baseline_hidden_dim, device=x.device),
            bl_c=torch.zeros(bs, self.baseline_hidden_dim, device=x.device),
            z_pres=torch.ones(bs, 1, device=x.device),
            z_where=torch.zeros(bs, 3, device=x.device),
            z_what=torch.zeros(bs, self.z_what_dim, device=x.device),
            z_what_loc=torch.zeros(bs, self.z_what_dim, device=x.device),
        )

        # KL divergence for each step
        kl = torch.zeros(bs, self.max_steps, device=x.device)

        # Store KL for pres, where, and what separately
        kl_pres = torch.zeros(bs, self.max_steps, device=x.device)
        kl_where = torch.zeros(bs, self.max_steps, device=x.device)
        kl_what = torch.zeros(bs, self.max_steps, device=x.device)

        # Baseline value for each step
        baseline_value = torch.zeros(bs, self.max_steps, device=x.device)

        # Log likelihood for each step, with shape (B, T):
        # log q(z_pres[t] | x, z_{<t}), but only for t <= n+1
        z_pres_likelihood = torch.zeros(bs, self.max_steps, device=x.device)

        # Baseline target for each step
        baseline_target = torch.zeros(bs, self.max_steps, device=x.device)

        # signal_mask (prev.z_pres) for each step
        mask_prev = torch.ones(bs, self.max_steps, device=x.device)

        # Mask (z_pres) for each step
        mask_curr = torch.ones(bs, self.max_steps, device=x.device)

        # Output canvas
        h = w = self.img_size
        ch = self.color_channels
        canvas = torch.zeros(bs, ch, h, w, device=x.device)


        # Save z_where to visualize bounding boxes
        all_z_where = torch.zeros(bs, self.max_steps, 3, device=x.device)
        all_z_what = torch.zeros(bs, self.max_steps, self.z_what_dim, device=x.device)

        step_image = []

        for t in range(self.max_steps):
            # This is the previous z_pres, so at step i=0 this mask is all 1s.
            # It is used to zero out all time steps after the first z_pres=0.
            # The first z_pres=0 is NOT masked.
            mask_prev[:, t] = state.z_pres.squeeze()
            
            # Do one inference step and save results
            result = self.inference_step(state, x)
            state = result['state']
            kl[:, t] = result['kl']
            kl_pres[:, t] = result['kl_pres']
            kl_where[:, t] = result['kl_where']
            kl_what[:, t] = result['kl_what']
            baseline_value[:, t] = result['baseline_value']
            z_pres_likelihood[:, t] = result['z_pres_likelihood']

            # Add KL at timestep t to baseline_target for timesteps 0 to t
            # At the end of the loop: baseline_target[t] = sum_{i=t}^T KL[i]
            for j in range(t + 1):
                baseline_target[:, j] += result['kl']
                
            # Decode z_what to object appearance
            sprite = self.decoder(state.z_what) 
           

            # Spatial-transform it to image with shape (B, 1, H, W)
            img = self.spatial_transf.forward(sprite, state.z_where)

            # Add to the output canvas, masking according to object presence
            # state.z_pres has shape (B, 1)
            canvas += img * state.z_pres[:, :, None, None]
            step_image.append(img * state.z_pres[:, :, None, None])
            ## shape of canvas [32, 3, 64, 64]


            # Presence mask for current time step
            mask_curr[:, t] = state.z_pres.squeeze(1)

            # Save z_where to visualize bounding boxes
            all_z_where[:, t] = state.z_where  # shape (B, 3)
            all_z_what[:, t] = state.z_what_loc

        # Clip canvas to [0, 1] (lose gradient where overlap)
        if self.likelihood == 'bernoulli':
            eps = 1e-5
            canvas = canvas.clamp(min=eps, max=1.0-eps)
            # canvas = canvas.clamp(min=0., max=1.)

        # Inferred number of objects in each image
        inferred_n = mask_curr.sum(1)   # shape (B,)

        # Output distribution p(x | z)
        output_dist = self.get_output_dist(canvas)

        # Data likelihood log p(x | z)
        likelihood_sep = output_dist.log_prob(x)

        # Sample from log p(x | z) with inferred z
        out_sample = output_dist.sample()

        # Sum over all data dimensions, resulting shape (B, )
        likelihood_sep = likelihood_sep.sum((1, 2, 3))
        # likelihood_sep = likelihood_sep.clamp(min=0)

        # Sum KL over time steps, resulting shape (B, )
        kl = kl.sum(1)


        # ELBO separated per sample, shape (B, )
        elbo_sep = likelihood_sep - kl

        data = {
            'elbo_sep': elbo_sep,
            'elbo': elbo_sep.mean(),
            'inferred_n': inferred_n,
            'data_likelihood': likelihood_sep,
            'recons': -likelihood_sep.mean(),
            'kl': kl,
            'kl_mean': kl.mean(),
            'kl_pres': kl_pres.sum(1).mean(),
            'kl_where': kl_where.sum(1).mean(),
            'kl_what': kl_what.sum(1).mean(),
            'out_mean': canvas,
            'out_sample': out_sample,
            'all_z_where': all_z_where,
            'baseline_target': baseline_target,
            'baseline_value': baseline_value,
            'mask_prev': mask_prev,
            'z_pres_likelihood': z_pres_likelihood,
            'step_image': step_image,
            'all_z_what': all_z_what,
            'z_pres': state.z_pres.mean()
        }

        return data


    def inference_step(self, prev, x):
        """
        Given previous (or initial) state and input image, predict the next
        inference step (next object).
        """

        bs = x.size(0)
        
        # Flatten the image
        x_flat = x.reshape(bs, -1)
        # Feed (x, z_{<t}) through the LSTM cell, get encoding h
        lstm_input = torch.cat(
            (x_flat, prev.z_where, prev.z_what, prev.z_pres), dim=1)
        h, c = self.lstm(lstm_input, (prev.h, prev.c))

        # Predictor presence and location from h
        z_pres_p, z_where_loc, z_where_scale = self.predictor(h)
        
        # If previous z_pres is 0, force z_pres to 0
        z_pres_p = z_pres_p * prev.z_pres
        
        # Numerical stability
        eps = 1e-2
        z_pres_p = z_pres_p.clamp(min=eps, max=1.0-eps)

        # sample z_pres
        z_pres_post = Bernoulli(z_pres_p)
        # print('z_pres_p', torch.min(z_pres_p))
        z_pres = z_pres_post.sample()

        # If previous z_pres is 0, then this z_pres should also be 0.
        # However, this is sampled from a Bernoulli whose probability is at
        # least eps. In the unlucky event that the sample is 1, we force this
        # to 0 as well.
        z_pres = z_pres * prev.z_pres

        # Likelihood: log q(z_pres[i] | x, z_{<i}) (if z_pres[i-1]=1, else 0)
        # Mask with prev.z_pres instead of z_pres, i.e. if already at the
        # previous step there was no object.
        z_pres_likelihood = z_pres_post.log_prob(z_pres) * prev.z_pres
        z_pres_likelihood = z_pres_likelihood.squeeze()  # shape (B,)

        # Sample z_where
        z_where_post = Normal(z_where_loc, z_where_scale)
        z_where = z_where_post.rsample()
        # Get object from image - shape (B, 1, Hobj, Wobj)
        obj = self.spatial_transf.inverse(x, z_where)

        # Predictor z_what
        z_what_loc, z_what_scale = self.encoder(obj)
        z_what_post = Normal(z_what_loc, z_what_scale)
        z_what = z_what_post.rsample()

        # Compute baseline for this z_pres:
        # b_i(z_{<i}) depending on previous step latent variables only.
        bl_h, bl_c = self.bl_lstm(lstm_input.detach(), (prev.bl_h, prev.bl_c))
        baseline_value = self.bl_regressor(bl_h).squeeze()  # shape (B,)

        # The baseline is not used if z_pres[t-1] is 0 (object not present in
        # the previous step). Mask it out to be on the safe side.
        baseline_value = baseline_value * prev.z_pres.squeeze()
        
        # KL for the current step, sum over data dimension: shape (B,)
        kl_pres = kl_divergence(
            z_pres_post,
            self.pres_prior.expand(z_pres_post.batch_shape)).sum(1)
        kl_where = kl_divergence(
            z_where_post,
            self.where_prior.expand(z_where_post.batch_shape)).sum(1)
        kl_what = kl_divergence(
            z_what_post,
            self.what_prior.expand(z_what_post.batch_shape)).sum(1)

        # When z_pres[i] is 0, zwhere and zwhat are not used -> set KL=0
        kl_where = kl_where * z_pres.squeeze()
        kl_what = kl_what * z_pres.squeeze()

        # When z_pres[i-1] is 0, zpres is not used -> set KL=0
        kl_pres = kl_pres * prev.z_pres.squeeze()
        
        kl = (kl_pres + kl_where + kl_what)

        # New state
        new_state = State(
            z_pres=z_pres,
            z_where=z_where,
            z_what=z_what,
            h=h,
            c=c,
            bl_c=bl_c,
            bl_h=bl_h,
            z_what_loc=z_what_loc,
            )

        out = {
            'state': new_state,
            'kl': kl,
            'kl_pres': kl_pres,
            'kl_where': kl_where,
            'kl_what': kl_what,
            'baseline_value': baseline_value,
            'z_pres_likelihood': z_pres_likelihood,
        }
        return out


    def sample_prior(self, n_imgs, **kwargs):

        # Sample from prior. Shapes:
        # z_pres:  (B, T)
        # z_what:  (B, T, z_what_dim)
        # z_where: (B, T, 3)
        z_pres = self.pres_prior.sample((n_imgs, self.max_steps))
        z_what = self.what_prior.sample((n_imgs, self.max_steps, self.z_what_dim))
        z_where = self.where_prior.sample((n_imgs, self.max_steps))

        # TODO This is only for visualization! Not real model samples
        # The prior of z_pres puts a lot of probability on n=0, which doesn't
        # lead to informative samples. Instead, generate half images with 1
        # object and half with 2.
        # z_pres.fill_(0.)
        # z_pres[:, 0].fill_(1.)
        # z_pres[n_imgs//2:, 1].fill_(1.)

        # If z_pres is sampled from the prior, make sure there are no ones
        # after a zero.
        for t in range(1, self.max_steps):
            z_pres[:, t] *= z_pres[:, t-1]  # if previous=0, this is also 0

        n_obj = z_pres.sum(1)

        # Decode z_what to object appearance
        sprites = self.decoder(z_what)

        # Spatial-transform them to images with shape (B*T, 1, H, W)
        z_where_ = z_where.view(n_imgs * self.max_steps, 3)  # shape (B*T, 3)
        imgs = self.spatial_transf.forward(sprites, z_where_)

        # Reshape images to (B, T, 1, H, W)
        h = w = self.img_size
        ch = self.color_channels
        imgs = imgs.view(n_imgs, self.max_steps, ch, h, w)

        # Make canvas by masking and summing over timesteps
        canvas = imgs * z_pres[:, :, None, None, None]
        canvas = canvas.sum(1)

        return canvas, z_where, n_obj