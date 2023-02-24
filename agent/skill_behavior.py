import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F

import utils
import agent.net_utils as common
from agent.mb_utils import stop_gradient

def get_feat_ac(seq):
  return torch.cat([seq['feat'], seq['skill']], dim=-1) 

class SkillActorCritic(nn.Module):
  def __init__(self, config, act_spec, tfstep, skill_dim, solved_meta=None, imagine_obs=False):
    super().__init__()
    self.cfg = config
    self.act_spec = act_spec
    self.tfstep = tfstep
    self._use_amp = (config.precision == 16)
    self.device = config.device

    self.imagine_obs = imagine_obs
    self.solved_meta = solved_meta
    self.skill_dim = skill_dim
    inp_size = config.rssm.deter 
    if config.rssm.discrete: 
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch
    
    inp_size += skill_dim
    self.actor = common.MLP(inp_size, act_spec.shape[0], **self.cfg.actor)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0 
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('skill_actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer('skill_critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = common.StreamNorm(**self.cfg.skill_reward_norm, device=self.device)

  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.cfg.imag_horizon
    with common.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        B,T , _ = start['deter'].shape
        if self.solved_meta is not None:
          img_skill = torch.from_numpy(self.solved_meta['skill']).repeat(B*T, 1).to(self.device)
        else:
          img_skill = F.one_hot(torch.randint(0, self.skill_dim, 
                                      size=(B*T,), device=self.device), num_classes=self.skill_dim).float()

        seq = world_model.imagine(self.actor, start, is_terminal, hor, skill_cond=img_skill)
        if self.imagine_obs:
          with torch.no_grad():
            seq['observation'] = world_model.heads['decoder'](seq['feat'].detach())['observation'].mean
        reward = reward_fn(seq)
        seq['reward'], mets1 = self.rewnorm(reward)
        mets1 = {f'skill_reward_{k}': v for k, v in mets1.items()}
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
    policy = self.actor(stop_gradient(get_feat_ac(seq)[:-2]))
    if self.cfg.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.cfg.actor_grad == 'reinforce':
      baseline = self._target_critic(get_feat_ac(seq)[:-2]).mean 
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
    elif self.cfg.actor_grad == 'both':
      baseline = self._target_critic(get_feat_ac(seq)[:-2]).mean 
      advantage = stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(stop_gradient(seq['action'][1:-1]))[:,:,None] * advantage
      mix = utils.schedule(self.cfg.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['skill_actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.cfg.actor_grad)
    ent = policy.entropy()[:,:,None]
    ent_scale = utils.schedule(self.cfg.skill_actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean() 
    metrics['skill_actor_ent'] = ent.mean()
    metrics['skill_actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    dist = self.critic(get_feat_ac(seq)[:-1])
    target = stop_gradient(target)
    weight = stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target)[:,:,None] * weight[:-1]).mean()
    metrics = {'skill_critic': dist.mean.mean() } 
    return critic_loss, metrics

  def target(self, seq):
    reward = seq['reward'] 
    disc = seq['discount'] 
    value = self._target_critic(get_feat_ac(seq)).mean 
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.cfg.discount_lambda,
        axis=0)
    metrics = {}
    metrics['skill_critic_slow'] = value.mean()
    metrics['skill_critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.cfg.slow_target:
      if self._updates % self.cfg.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.cfg.slow_target_fraction)
        for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1 

class MetaCtrlAC(nn.Module):
  def __init__(self, config, skill_dim, tfstep, skill_executor, frozen_skills=False, skill_len=1):
    super().__init__()
    self.cfg = config
    self.skill_dim = skill_dim
    self.tfstep = tfstep
    self.skill_executor = skill_executor
    self._use_amp = (config.precision == 16)
    self.device = config.device
    
    inp_size = config.rssm.deter 
    if config.rssm.discrete: 
      inp_size += config.rssm.stoch * config.rssm.discrete
    else:
      inp_size += config.rssm.stoch

    actor_config = {'layers': 4, 'units': 400, 'norm': 'none', 'dist': 'trunc_normal' }
    actor_config['dist'] = 'onehot' 
    self.actor = common.MLP(inp_size, skill_dim, **actor_config)
    self.critic = common.MLP(inp_size, (1,), **self.cfg.critic)
    if self.cfg.slow_target:
      self._target_critic = common.MLP(inp_size, (1,), **self.cfg.critic)
      self._updates = 0 
    else:
      self._target_critic = self.critic

    self.termination = False
    self.skill_len = skill_len

    self.selector_opt = common.Optimizer('selector_actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.executor_opt = common.Optimizer('executor_actor', self.skill_executor.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    self.critic_opt = common.Optimizer('selector_critic', self.critic.parameters(), **self.cfg.critic_opt, use_amp=self._use_amp)
    self.rewnorm = common.StreamNorm(**self.cfg.reward_norm, device=self.device)

  def update(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.cfg.imag_horizon
    with common.RequiresGrad(self.actor):
      with common.RequiresGrad(self.skill_executor.actor):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
          seq = self.selector_imagine(world_model, self.actor, start, is_terminal, hor)
          reward = reward_fn(seq)
          seq['reward'], mets1 = self.rewnorm(reward)
          mets1 = {f'reward_{k}': v for k, v in mets1.items()}
          target, mets2 = self.target(seq)
          high_actor_loss, low_actor_loss, mets3 = self.actor_loss(seq, target)
        metrics.update(self.selector_opt(high_actor_loss, self.actor.parameters()))
        metrics.update(self.executor_opt(low_actor_loss, self.skill_executor.actor.parameters()))
    with common.RequiresGrad(self.critic):
      with torch.cuda.amp.autocast(enabled=self._use_amp):
        seq = {k: stop_gradient(v) for k,v in seq.items()}
        critic_loss, mets4 = self.critic_loss(seq, target)
      metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target): 
    self.tfstep = 0 
    metrics = {}
    skill = stop_gradient(seq['skill'])
    action = stop_gradient(seq['action'])
    policy = self.actor(stop_gradient(seq['feat'][:-2]))
    low_inp = stop_gradient(torch.cat([seq['feat'][:-2], skill[:-2]], dim=-1))
    low_policy = self.skill_executor.actor(low_inp)
    if self.cfg.actor_grad == 'dynamics':
      low_objective = target[1:]
    
    ent_scale = utils.schedule(self.cfg.actor_ent, self.tfstep)
    weight = stop_gradient(seq['weight'])

    low_ent = low_policy.entropy()[:,:,None]
    high_ent = policy.entropy()[:,:,None]

    baseline = self._target_critic(seq['feat'][:-2]).mean 
    advantage = stop_gradient(target[1:] - baseline)
    log_probs = policy.log_prob(skill[1:-1])[:,:,None]

    # Note: this is impactful only if skill_len > 1. In Choreographer we fixed skill_len to 1
    indices = torch.arange(0, log_probs.shape[0], step=self.skill_len, device=self.device)
    advantage = torch.index_select(advantage, 0, indices)
    log_probs = torch.index_select(log_probs, 0, indices)
    high_ent = torch.index_select(high_ent, 0, indices)
    high_weight = torch.index_select(weight[:-2], 0, indices)

    high_objective =  log_probs * advantage
    if getattr(self, 'reward_smoothing', False):
      high_objective *= 0   
      low_objective *= 0

    high_objective += ent_scale * high_ent
    high_actor_loss = -(high_weight * high_objective).mean() 
    low_actor_loss  = -(weight[:-2] * low_objective).mean() 
    
    metrics['high_actor_ent'] = high_ent.mean()
    metrics['low_actor_ent'] = low_ent.mean()
    metrics['skills_updated'] = len(torch.unique(torch.argmax(skill, dim=-1)))
    return high_actor_loss, low_actor_loss, metrics

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

  def selector_imagine(self, wm, policy, start, is_terminal, horizon, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = wm.rssm.get_feat(start)
    inp = start['feat']
    fake_skill = policy(inp).mean
    fake_action = self.skill_executor.actor(torch.cat([inp, fake_skill], dim=-1)).mean
    B, _ = fake_action.shape
    start['skill'] = torch.zeros_like(fake_skill, device=wm.device) 
    start['action'] = torch.zeros_like(fake_action, device=wm.device) 
    seq = {k: [v] for k, v in start.items()}
    for h in range(horizon):
      inp = stop_gradient(seq['feat'][-1]) 
      if h % self.skill_len == 0:
        skill = policy(inp)
        if not eval_policy:
          skill = skill.sample()
        else:
          skill = skill.mode()

      executor_inp = stop_gradient(torch.cat([inp, skill], dim=-1)) 
      action = self.skill_executor.actor(executor_inp)
      action = action.sample() if not eval_policy else action.mean 
      state = wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = wm.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat, 'skill' : skill,}.items():
        seq[key].append(value)
    # shape will be (T, B, *DIMS)
    seq = {k: torch.stack(v, 0) for k, v in seq.items()}
    if 'discount' in wm.heads:
      disc = wm.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal) 
        true_first *= wm.cfg.discount
        disc = torch.cat([true_first[None], disc[1:]], 0)
    else:
      disc = wm.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=wm.device)
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = torch.cumprod(
        torch.cat([torch.ones_like(disc[:1], device=wm.device), disc[:-1]], 0), 0)
    return seq

