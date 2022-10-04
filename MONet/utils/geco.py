# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import torch


class GECO():

    def __init__(self, alpha,
                beta_goal, beta_step_size,
                gamma_goal, gamma_step_size,
                beta_init, beta_min, beta_speedup,
                gamma_init, gamma_min, gamma_speedup):
        self.err_ema = None
        self.alpha = alpha
        self.beta_goal = beta_goal
        self.beta_step_size = beta_step_size
        self.gamma_goal = gamma_goal
        self.gamma_step_size = gamma_step_size
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e10)
        self.beta_speedup = beta_speedup
        self.gamma = torch.tensor(gamma_init)
        self.gamma_min = torch.tensor(gamma_min)
        self.gamma_max = torch.tensor(1e10)
        self.gamma_speedup = gamma_speedup

    def to_cuda(self):
        self.beta = self.beta.cuda()
        self.gamma = self.gamma.cuda()
        if self.err_ema is not None:
            self.err_ema = self.err_ema.cuda()

    def loss(self, err, kl_l, kl_m):
        # Compute loss with current beta
        loss = err + self.beta * kl_l + self.gamma * kl_m
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            ## calculate exponential moving average of error (0.1*current error + 0.99*prev_ema)
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0-self.alpha)*err + self.alpha*self.err_ema
            
            beta_constraint = (self.beta_goal - self.err_ema)
            if self.beta_speedup is not None and beta_constraint.item() > 0: ## only speedu when increase beta?
                beta_factor = torch.exp(self.beta_speedup * self.beta_step_size * beta_constraint)
            else:
                beta_factor = torch.exp(self.beta_step_size * beta_constraint)
            self.beta = (beta_factor * self.beta).clamp(self.beta_min, self.beta_max)
            
            gamma_constraint = (self.gamma_goal - self.err_ema)
            if self.gamma_speedup is not None and gamma_constraint.item() > 0:
                gamma_factor = torch.exp(self.gamma_speedup * self.gamma_step_size * gamma_constraint)
            else:
                gamma_factor = torch.exp(self.gamma_step_size * gamma_constraint)
            self.gamma = (gamma_factor * self.gamma).clamp(self.gamma_min, self.gamma_max)

        return loss
