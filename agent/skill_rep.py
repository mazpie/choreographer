import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import agent.net_utils as common
from agent.mb_utils import *
from agent.skill_behavior import *

class SkillModule(nn.Module):
  def __init__(self, config, skill_dim, vq_ema=True, code_dim=16, code_resampling=True, resample_every=200):
    super().__init__()
    self.cfg = config
    self._use_amp = (config.precision == 16)
    self.device = config.device
    self._updates = 0
    self.code_resampling = code_resampling
    self.resample_every = resample_every
    
    self.vq_ema = vq_ema
    self.code_dim = code_dim

    self.skill_dim = skill_dim
    inp_size = config.rssm.deter 
    if config.rssm.discrete: 
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch

    mlp_config = {**self.cfg.reward_head}
    mlp_config['norm'] = 'layer'
    self.comit_coef = 1e-4
    self.vq_coef    = 0.05
    self.skill_encoder = common.MLP(config.rssm.deter, (code_dim), **mlp_config)
    self.skill_decoder = common.MLP(code_dim, (config.rssm.deter), **mlp_config)
    self.emb = NearestEmbedEMA(skill_dim, code_dim) if self.vq_ema else NearestEmbed(skill_dim, code_dim)
    self.all_params = list(self.emb.parameters()) + list(self.skill_encoder.parameters()) + list(self.skill_decoder.parameters())

    opt_config = { **self.cfg.actor_opt }

    self.latent_info = dict()
    self.skill_opt = common.Optimizer('skill', self.all_params, **opt_config, use_amp=self._use_amp)

  def update(self, seq,):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    seq = {k: flatten(v) for k, v in seq.items()}
    with common.RequiresGrad(self):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        skill_loss, metrics = self.skill_loss(seq)
      metrics.update(self.skill_opt(skill_loss, self.all_params))
    self._updates += 1
    if self._updates % self.resample_every == 0:
      with torch.no_grad():
        z_e = self.skill_encoder(seq['deter']).mean
        self.latent_info.update(self.emb.kmeans(z_e, update=self.code_resampling))
    metrics.update(self.latent_info)
    return metrics

  def skill_loss(self, seq):
    z_e = self.skill_encoder(seq['deter']).mean

    if self.vq_ema:
      emb, _ = self.emb(z_e, training=True)
      recon = self.skill_decoder(emb).mean

      rec_loss = F.mse_loss(recon, seq['deter'])
      commit_loss = F.mse_loss(z_e, emb.detach())

      loss = rec_loss + self.comit_coef*commit_loss
      return loss, {'rec_loss': rec_loss, 'commit_loss': commit_loss}
    else:
      z_q, _ = self.emb(z_e, weight_sg=True)
      emb, _ = self.emb(z_e.detach())
      recon = self.skill_decoder(z_q).mean

      rec_loss = F.mse_loss(recon, seq['deter'])
      vq_loss = F.mse_loss(emb, z_e.detach())
      commit_loss = F.mse_loss(z_e, emb.detach())

      loss = rec_loss + self.vq_coef*vq_loss + self.comit_coef*commit_loss
      return loss, {'rec_loss': rec_loss, 'vq_loss' : vq_loss, 'commit_loss': commit_loss}

class NearestEmbedFunc(torch.autograd.Function):
# Adapted from: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/nearest_embed.py
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None

def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)

class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)

class NearestEmbedEMA(nn.Module):
    # Inspired by : https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/nearest_embed.py
    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('prev_cluster', torch.zeros(n_emb))

    def forward(self, x, *args, training=False, **kwargs):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """

        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(
                self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(
            0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        if training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) ==
                          latent_indices.view(1, -1)).type_as(x.data)
            
            n_idx_choice = emb_onehot.sum(0)
            self.prev_cluster.data.add_(n_idx_choice)
            n_idx_choice[n_idx_choice == 0] = 1
            
            if num_arbitrary_dims:
              flatten = x.permute(
                  1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)
            else:
              flatten = x.permute(1, 0).contiguous().view(x.shape[1], -1) 

            self.cluster_size.data.mul_(self.decay).add_(
                n_idx_choice, alpha=1 - self.decay)
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_emb * self.eps) * n

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)

        result = x + (result - x).detach()

        return result, argmin
    
    def kmeans(self, x, update=False):
        metrics = dict()
        metrics['unused_codes']  = torch.sum(self.prev_cluster == 0.)
        updated = 0
        batch_size = x.shape[0]
        if update:
          with torch.no_grad():
            dims = list(range(len(x.size())))
            x_expanded = x.unsqueeze(-1)
            num_arbitrary_dims = len(dims) - 2

            for idx, eq in enumerate(self.prev_cluster):
              if eq == 0:
                if num_arbitrary_dims:
                  emb_expanded = self.weight.view(
                      self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
                else:
                    emb_expanded = self.weight

                dist = torch.norm(x_expanded - emb_expanded, 2, 1)
                min_dist, argmin = dist.min(-1)

                probs = min_dist / (torch.sum(min_dist, dim=0, keepdim=True) + 1e-6)
                if probs.sum() == 0:
                  break
                x_idx = torch.multinomial(probs, 1)
                self.weight.data[:, idx].copy_(x[x_idx].squeeze())
                self.embed_avg.data[:, idx].copy_(x[x_idx].squeeze())

                updated += 1

        metrics['resampled_codes'] = updated
        self.prev_cluster.data.mul_(0.)
        return metrics
            