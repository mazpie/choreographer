import torch.nn as nn
import torch

from collections import OrderedDict
import numpy as np
from dm_env import specs

import utils
import agent.net_utils as common
from agent.mb_utils import *
from agent.skill_behavior import *
from agent.skill_rep import *

class ChoreoAgent(nn.Module):
  def __init__(self, name, cfg, obs_space, act_spec, **kwargs):
    super().__init__()
    self.name = name
    self.cfg = cfg
    self.cfg.update(**kwargs)
    self.obs_space = obs_space
    self.act_spec = act_spec
    self.tfstep = None 
    self._use_amp = (cfg.precision == 16)
    self.device = cfg.device
    self.act_dim = act_spec.shape[0]
    
    # World model
    self.wm = WorldModel(cfg, obs_space, self.act_dim, self.tfstep)
    self.wm.recon_skills = True

    # Exploration
    self._env_behavior = ActorCritic(cfg, self.act_spec, self.tfstep)
    self.lbs = common.MLP(self.wm.inp_size, (1,), **self.cfg.reward_head).to(self.device)
    self.lbs_opt = common.Optimizer('lbs', self.lbs.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
    self.lbs.train()

    # Skills
    self.skill_dim = kwargs['skill_dim']
    self.skill_pbe = utils.PBE(utils.RMS(self.device), kwargs['knn_clip'], kwargs['knn_k'], kwargs['knn_avg'], kwargs['knn_rms'], self.device) 

    self._skill_behavior = SkillActorCritic(self.cfg, self.act_spec, self.tfstep, self.skill_dim,)
    self.skill_module = SkillModule(self.cfg, self.skill_dim, code_dim=kwargs['code_dim'], code_resampling=kwargs['code_resampling'], resample_every=kwargs['resample_every'])
    self.wm.skill_module = self.skill_module

    # Adaptation
    self.num_init_frames = kwargs['num_init_frames']
    self.update_task_every_step = self.update_skill_every_step = kwargs['update_skill_every_step']
    self.is_ft = False

    # Common
    self.to(self.device)
    self.requires_grad_(requires_grad=False)

  def init_meta(self):
      return self.init_meta_discrete()

  def get_meta_specs(self):
      return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

  def init_meta_discrete(self):
      skill = np.zeros(self.skill_dim, dtype=np.float32)
      skill[np.random.choice(self.skill_dim)] = 1.0
      meta = OrderedDict()
      meta['skill'] = skill
      return meta

  def update_meta(self, meta, global_step, time_step):
      if global_step % self.update_skill_every_step == 0:
          return self.init_meta()
      return meta

  def finetune_mode(self):
    self.is_ft = True
    self.reward_smoothing = True
    self.cfg.actor_ent = 1e-4
    self.cfg.skill_actor_ent = 1e-4
    self._env_behavior = MetaCtrlAC(self.cfg, self.skill_dim, self.tfstep, self._skill_behavior, 
                                          frozen_skills=self.cfg.freeze_skills, skill_len=int(1)).to(self.device)
    self._env_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8}, device=self.device)
    self._skill_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8}, device=self.device)

  def act(self, obs, meta, step, eval_mode, state):
    # Infer current state
    obs = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
    meta = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in meta.items()}
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
    else:
      latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
    feat = self.wm.rssm.get_feat(latent)

    # PT stage
    # -> LBS Exploration 
    if not self.is_ft:
      if eval_mode:
        actor = self._env_behavior.actor(feat)
        action = actor.mean
      else:
        actor = self._env_behavior.actor(feat)
        action = actor.sample()
      new_state = (latent, action)
      return action.cpu().numpy()[0], new_state

    # else (IS FT)
    is_adaptation = self.is_ft and (step >= self.num_init_frames // self.cfg.action_repeat)

    # Is FT AND step > num_init_frames and reward was already found 
    # -> use the meta-controller
    if is_adaptation and (not self.reward_smoothing):
      if eval_mode:
        skill = self._env_behavior.actor(feat)
        skill = skill.mode()
        action = self._skill_behavior.actor(torch.cat([feat, skill], dim=-1))
        action = action.mean
      else:
        skill = self._env_behavior.actor(feat)
        skill = skill.sample()
        action = self._skill_behavior.actor(torch.cat([feat, skill], dim=-1))
        action = action.sample()
      new_state = (latent, action)
      return action.cpu().numpy()[0], new_state

    # Cases:
    # 1 - is not adaptation (independently from the reward smoothing)
    # 2 - is adaptation and the reward smoothing is still active
    # -> follow randomly sampled meta['skill']
    else:
      skill = meta['skill']
      inp = torch.cat([feat, skill], dim=-1)
      if eval_mode:
        actor = self._skill_behavior.actor(inp)
        action = actor.mean
      else:
        actor = self._skill_behavior.actor(inp)
        action = actor.sample()
      new_state = (latent, action)
      return action.cpu().numpy()[0], new_state

  def pbe_reward_fn(self, seq):
    rep = seq['deter']
    B, T, _ = rep.shape
    reward = self.skill_pbe(rep.reshape(B*T, -1), cdist=True, apply_log=False).reshape(B, T, 1)
    return reward.detach()

  def code_reward_fn(self, seq):
    T, B, _ = seq['skill'].shape
    skill_target = seq['skill'].reshape(T*B, -1)
    vq_skill = skill_target @ self.skill_module.emb.weight.T
    state_pred = self.skill_module.skill_decoder(vq_skill).mean.reshape(T, B, -1)
    reward = -torch.norm(state_pred - seq['deter'], p=2, dim=-1).reshape(T, B, 1)  
    return reward

  def skill_mi_fn(self, seq):
    ce_rw   = self.code_reward_fn(seq)
    ent_rw  = self.pbe_reward_fn(seq)
    return ent_rw + ce_rw

  def update_lbs(self, outs):
    metrics = dict()
    B, T, _ = outs['feat'].shape
    feat, kl = outs['feat'].detach(), outs['kl'].detach()
    feat = feat.reshape(B*T, -1)
    kl = kl.reshape(B*T, -1)

    loss = -self.lbs(feat).log_prob(kl).mean()
    metrics.update(self.lbs_opt(loss, self.lbs.parameters()))
    metrics['lbs_loss'] = loss.item()
    return metrics

  def update_behavior(self, state=None, outputs=None, metrics={}, data=None):
    if outputs is not None:
      post = outputs['post']
      is_terminal = outputs['is_terminal']
    else:
      data = self.wm.preprocess(data)
      embed = self.wm.encoder(data)
      post, _ = self.wm.rssm.observe(
          embed, data['action'], data['is_first'])
      is_terminal = data['is_terminal']
    #
    start = {k: stop_gradient(v) for k,v in post.items()}
    # Train skill (module + AC)
    start['feat'] = stop_gradient(self.wm.rssm.get_feat(start))
    metrics.update(self.skill_module.update(start))
    metrics.update(self._skill_behavior.update(
        self.wm, start, is_terminal, self.skill_mi_fn))
    return start, metrics

  def update_wm(self, data, step):
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    outputs['is_terminal'] = data['is_terminal']
    metrics.update(mets)
    return state, outputs, metrics

  def update(self, data, step):
    # Train WM
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    metrics.update(mets)
    start = outputs['post']
    start = {k: stop_gradient(v) for k,v in start.items()}
    if not self.is_ft:
      # LBS exploration 
      with common.RequiresGrad(self.lbs):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            metrics.update(self.update_lbs(outputs))
      reward_fn = lambda seq: self.lbs(seq['feat']).mean
      metrics.update(self._env_behavior.update(
          self.wm, start, data['is_terminal'], reward_fn))

      # Train skill (module + AC)
      start['feat'] = stop_gradient(self.wm.rssm.get_feat(start))
      metrics.update(self.skill_module.update(start))
      metrics.update(self._skill_behavior.update(
          self.wm, start, data['is_terminal'], self.skill_mi_fn))
    else:
      self.reward_smoothing =  self.reward_smoothing and (not (data['reward'] > 1e-4).any())
      self._env_behavior.reward_smoothing = self.reward_smoothing 

      # Train task AC
      if not self.reward_smoothing:
        reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean 
        metrics.update(self._env_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
    return state, metrics

  def init_from(self, other):
      # WM 
      print(f"Copying the pretrained world model")
      utils.hard_update_params(other.wm.rssm, self.wm.rssm)
      utils.hard_update_params(other.wm.encoder, self.wm.encoder)
      utils.hard_update_params(other.wm.heads['decoder'], self.wm.heads['decoder'])

      # Skill
      print(f"Copying the pretrained skill modules")
      utils.hard_update_params(other._skill_behavior.actor, self._skill_behavior.actor)
      utils.hard_update_params(other.skill_module, self.skill_module)
      if getattr(self.skill_module, 'emb', False):
        self.skill_module.emb.weight.data.copy_(other.skill_module.emb.weight.data)

  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      report[f'openl_{name}'] = self.wm.video_pred(data, key)
    return report