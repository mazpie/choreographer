import torch.nn as nn
import torch

import utils
import agent.net_utils as common
import numpy as np

def stop_gradient(x):
  return x.detach()

class WorldModel(nn.Module):
  def __init__(self, config, obs_space, act_dim, tfstep):
    super().__init__()
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.shapes = shapes
    self.cfg = config
    self.device = config.device
    self.tfstep = tfstep
    self.encoder = common.Encoder(shapes, **config.encoder)
    # Computing embed dim
    with torch.no_grad():
      zeros = {k: torch.zeros( (1,) + v) for k, v in shapes.items()}
      outs = self.encoder(zeros)
      embed_dim = outs.shape[1]
    self.embed_dim = embed_dim
    self.rssm = common.EnsembleRSSM(**config.rssm, action_dim=act_dim, embed_dim=embed_dim, device=self.device)
    self.heads = {}
    self._use_amp = (config.precision == 16)
    inp_size = config.rssm.deter 
    if config.rssm.discrete: 
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch
    self.inp_size = inp_size
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder, embed_dim=inp_size)
    self.heads['reward'] = common.MLP(inp_size, (1,), **config.reward_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.grad_heads = config.grad_heads
    self.heads = nn.ModuleDict(self.heads)
    self.model_opt = common.Optimizer('model', self.parameters(), **config.model_opt, use_amp=self._use_amp)

  def update(self, data, state=None):
    with common.RequiresGrad(self):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        model_loss, state, outputs, metrics = self.loss(data, state)
      metrics.update(self.model_opt(model_loss, self.parameters())) 
    return state, outputs, metrics

  def loss(self, data, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.cfg.kl)
    assert len(kl_loss.shape) == 0 or (len(kl_loss.shape) == 1 and kl_loss.shape[0] == 1), kl_loss.shape
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.grad_heads)
      inp = feat if grad_head else stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = dist.log_prob(data[key]) 
        likes[key] = like
        losses[key] = -like.mean()
    model_loss = sum(
        self.cfg.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon, skill_cond=None, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    inp = start['feat'] if skill_cond is None else torch.cat([start['feat'], skill_cond], dim=-1)
    start['action'] = torch.zeros_like(policy(inp).mean, device=self.device) 
    seq = {k: [v] for k, v in start.items()}
    if skill_cond is not None: seq['skill'] = [skill_cond]
    for _ in range(horizon):
      inp = seq['feat'][-1] if skill_cond is None else torch.cat([seq['feat'][-1], skill_cond], dim=-1)
      action = policy(stop_gradient(inp)).sample() if not eval_policy else policy(stop_gradient(inp)).mean
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
      if skill_cond is not None: seq['skill'].append(skill_cond)
    # shape will be (T, B, *DIMS)
    seq = {k: torch.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal) 
        true_first *= self.cfg.discount
        disc = torch.cat([true_first[None], disc[1:]], 0)
    else:
      disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = torch.cumprod(
        torch.cat([torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0), 0)
    return seq

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype in [np.uint8, torch.uint8]:
        value = value / 255.0 - 0.5 
      obs[key] = value
    obs['reward'] = {
        'identity': nn.Identity(),
        'sign': torch.sign,
        'tanh': torch.tanh,
    }[self.cfg.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].float() 
    obs['discount'] *= self.cfg.discount
    return obs

  def video_pred(self, data, key, nvid=8):
    decoder = self.heads['decoder'] # B, T, C, H, W
    truth = data[key][:nvid] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:nvid, :5], data['action'][:nvid, :5], data['is_first'][:nvid, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mean[:nvid]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:nvid, 5:], init)
    prior_recon = decoder(self.rssm.get_feat(prior))[key].mean
    model = torch.clip(torch.cat([recon[:, :5] + 0.5, prior_recon + 0.5], 1), 0, 1)
    error = (model - truth + 1) / 2

    if getattr(self, 'recon_skills', False):
      B, T, _ = prior['deter'].shape
      z_e = self.skill_module.skill_encoder(prior['deter'].reshape(B*T, -1)).mean
      z_q, _ = self.skill_module.emb(z_e, weight_sg=True)
      latent_skills = z_q.reshape(B, T, -1)
      
      x = deter = self.skill_module.skill_decoder(latent_skills).mean

      stats = self.rssm._suff_stats_ensemble(x)
      index = torch.randint(0, self.rssm._ensemble, ()) 
      stats = {k: v[index] for k, v in stats.items()}
      dist = self.rssm.get_dist(stats)
      stoch = dist.sample() 
      prior = {'stoch': stoch, 'deter': deter, **stats}
      skill_recon = decoder(self.rssm.get_feat(prior))[key].mean
      error = torch.clip(torch.cat([recon[:, :5] + 0.5, skill_recon + 0.5], 1), 0, 1)

    video = torch.cat([truth, model, error], 3)
    B, T, C, H, W = video.shape
    return video 


class ActorCritic(nn.Module):
  def __init__(self, config, act_spec, tfstep):
    super().__init__()
    self.cfg = config
    self.act_spec = act_spec
    self.tfstep = tfstep
    self._use_amp = (config.precision == 16)
    self.device = config.device
    
    inp_size = config.rssm.deter 
    if config.rssm.discrete: 
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0 
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer('critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = common.StreamNorm(**self.cfg.reward_norm, device=self.device)

  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.cfg.imag_horizon
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = world_model.imagine(self.actor, start, is_terminal, hor)
        reward = reward_fn(seq)
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'reward_{k}': v for k, v in mets1.items()}
        target, mets2 = self.target(seq)
        actor_loss, mets3 = self.actor_loss(seq, target)
      metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target): #, step):
    self.tfstep = 0 
    metrics = {}
    policy = self.actor(stop_gradient(seq['feat'][:-2]))
    if self.cfg.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.cfg.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mean 
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
    elif self.cfg.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mean
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
      mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.cfg.actor_grad)
    ent = policy.entropy()[:,:,None]
    ent_scale = utils.schedule(self.cfg.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean() 
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    dist = self.critic(seq['feat'][:-1])
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target)[:,:,None] * weight[:-1]).mean()
    metrics = {'critic': dist.mean.mean() } 
    return critic_loss, metrics

  def target(self, seq):
    reward = seq['reward'] 
    disc = seq['discount'] 
    value = self._target_critic(seq['feat']).mean 
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1 