import re

import numpy as np

import utils
import torch.nn as nn
import torch
import torch.distributions as D
import torch.nn.functional as F

# Some functions are inspired by: https://github.com/jsikyoon/dreamer-torch

class SampleDist:
    def __init__(self, dist: D.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.rsample()

class OneHotDist(D.OneHotCategorical):
  def __init__(self, logits=None, probs=None):
    super().__init__(logits=logits, probs=probs)

  def mode(self):
    _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
    return _mode.detach() + super().logits - super().logits.detach()

  def sample(self, sample_shape=(), seed=None):
    if seed is not None:
      raise ValueError('need to check')
    sample = super().sample(sample_shape)
    probs = super().probs
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += probs - probs.detach() # ST-gradients
    return sample

class BernoulliDist(D.Bernoulli):
  def __init__(self, logits=None, probs=None):
    super().__init__(logits=logits, probs=probs)

  def sample(self, sample_shape=(), seed=None):
    if seed is not None:
      raise ValueError('need to check')
    sample = super().sample(sample_shape)
    probs = super().probs
    while len(probs.shape) < len(sample.shape):
      probs = probs[None]
    sample += probs - probs.detach() # ST-gradients
    return sample

def static_scan_for_lambda_return(fn, inputs, start):
  last = start
  indices = range(inputs[0].shape[0])
  indices = reversed(indices)
  flag = True
  for index in indices:
    inp = lambda x: (_input[x].unsqueeze(0) for _input in inputs)
    last = fn(last, *inp(index))
    if flag:
      outputs = last
      flag = False
    else:
      outputs = torch.cat([last, outputs], dim=0) 
  return outputs

def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  #assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * torch.ones_like(reward, device=reward.device)
  dims = list(range(len(reward.shape)))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = reward.permute(dims)
    value = value.permute(dims)
    pcont = pcont.permute(dims)
  if bootstrap is None:
    bootstrap = torch.zeros_like(value[-1], device=reward.device)
  if len(bootstrap.shape) < len(value.shape):
    bootstrap = bootstrap[None]
  next_values = torch.cat([value[1:], bootstrap], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = static_scan_for_lambda_return(
      lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
      (inputs, pcont), bootstrap)
  if axis != 0:
    returns = returns.permute(dims)
  return returns

def static_scan(fn, inputs, start, reverse=False, unpack=False):
  last = start
  indices = range(inputs[0].shape[0])
  flag = True
  for index in indices:
    inp = lambda x: (_input[x] for _input in inputs)
    if unpack:
      last = fn(last, inputs[0][index]) 
    else:
      last = fn(last, inp(index)) 
    if flag:
      if type(last) == type({}):
        outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
      else:
        outputs = []
        for _last in last:
          if type(_last) == type({}):
            outputs.append({key: value.clone().unsqueeze(0) for key, value in _last.items()})
          else:
            outputs.append(_last.clone().unsqueeze(0))
      flag = False
    else:
      if type(last) == type({}):
        for key in last.keys():
          outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
      else:
        for j in range(len(outputs)):
          if type(last[j]) == type({}):
            for key in last[j].keys():
              outputs[j][key] = torch.cat([outputs[j][key],
                  last[j][key].unsqueeze(0)], dim=0)
          else:
            outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
  if type(last) == type({}):
    outputs = [outputs]
  return outputs

class EnsembleRSSM(nn.Module):
  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act=nn.ELU, norm='none', std_act='softplus', min_std=0.1, action_dim=None, embed_dim=1536, device='cuda'):
    super().__init__()
    assert action_dim is not None 
    self.device = device
    self._embed_dim = embed_dim
    self._action_dim = action_dim
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = act()
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self._cell = GRUCell(self._hidden, self._deter, norm=True, device=self.device)

    if discrete:
      inp_dim = stoch * discrete + action_dim
    else:
      inp_dim = stoch + action_dim
    self._img_in = nn.Sequential(nn.Linear(inp_dim, hidden), NormLayer(norm, hidden))

    self._ensemble_img_out = nn.ModuleList([ nn.Sequential(nn.Linear(deter, hidden), NormLayer(norm, hidden)) for _ in range(ensemble)])
    if discrete:
      self._ensemble_img_dist = nn.ModuleList([ nn.Linear(hidden, stoch*discrete) for _ in range(ensemble)])
      self._obs_dist = nn.Linear(hidden, stoch*discrete)
    else:
      self._ensemble_img_dist = nn.ModuleList([ nn.Linear(hidden, 2*stoch) for _ in range(ensemble)])
      self._obs_dist = nn.Linear(hidden, 2*stoch)

    self._obs_out = nn.Sequential(nn.Linear(deter + embed_dim, hidden), NormLayer(norm, hidden))

  def initial(self, batch_size):
    if self._discrete:
      state = dict(
          logit=torch.zeros([batch_size, self._stoch, self._discrete], device=self.device), 
          stoch=torch.zeros([batch_size, self._stoch, self._discrete], device=self.device), 
          deter=self._cell.get_initial_state(None, batch_size)) 
    else:
      state = dict(
          mean=torch.zeros([batch_size, self._stoch], device=self.device), 
          std=torch.zeros([batch_size, self._stoch], device=self.device),
          stoch=torch.zeros([batch_size, self._stoch], device=self.device), 
          deter=self._cell.get_initial_state(None, batch_size)) 
    return state

  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    post, prior = static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (swap(action), swap(embed), swap(is_first)), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = static_scan(self.img_step, [action], state, unpack=True)[0] 
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch'] 
    if self._discrete:
      shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
      stoch = stoch.reshape(shape)
    return torch.cat([stoch, state['deter']], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      dist = D.Independent(OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      dist = D.MultivariateNormal(mean, torch.eye(std))
    return dist

  def obs_step(self, prev_state, prev_action, embed, is_first, should_sample=True):
    prev_state = { k: torch.einsum('b,b...->b...', 1.0 - is_first.float(), x) for k, x in prev_state.items()}
    prev_action = torch.einsum('b,b...->b...', 1.0 - is_first.float(), prev_action)
    #
    prior = self.img_step(prev_state, prev_action, should_sample)
    x = torch.cat([prior['deter'], embed], -1)
    x = self._obs_out(x)
    x = self._act(x)
    stats = self._suff_stats_layer('_obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if should_sample else None 
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = prev_state['stoch'] 
    if self._discrete:
      shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
      prev_stoch = prev_stoch.reshape(shape) 
    x = torch.cat([prev_stoch, prev_action], -1)
    x = self._img_in(x)
    x = self._act(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # It's wrapped in a list.
    stats = self._suff_stats_ensemble(x)
    index = torch.randint(0, self._ensemble, ()) 
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else None 
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self._ensemble_img_out[k](inp)
      x = self._act(x)
      stats.append(self._suff_stats_layer('_ensemble_img_dist', x, k=k))
    stats = {
        k: torch.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x, k=None):
    layer = getattr(self, name)
    if k is not None:
      layer = layer[k]
    x = layer(x)
    if self._discrete:
      logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      mean, std = torch.chunk(x, 2, -1)
      std = {
          'softplus': lambda: F.softplus(std),
          'sigmoid': lambda: torch.sigmoid(std),
          'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = D.kl_divergence
    sg = lambda x: {k: v.detach() for k, v in x.items()} 
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    dtype = post['stoch'].dtype
    device = post['stoch'].device
    free_tensor = torch.tensor([free], dtype=dtype, device=device)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = torch.maximum(value, free_tensor).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = torch.maximum(value_lhs.mean(), free_tensor)
        loss_rhs = torch.maximum(value_rhs.mean(), free_tensor)
      else:
        loss_lhs = torch.maximum(value_lhs, free_tensor).mean()
        loss_rhs = torch.maximum(value_rhs, free_tensor).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class Encoder(nn.Module):
  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act=nn.ELU, norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    super().__init__()
    self.shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Encoder CNN inputs:', list(self.cnn_keys))
    print('Encoder MLP inputs:', list(self.mlp_keys))
    self._act = act()
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

    if len(self.cnn_keys) > 0:
      self._conv_model = []
      for i, kernel in enumerate(self._cnn_kernels):
        if i == 0:
          prev_depth = 3
        else:
          prev_depth = 2 ** (i-1) * self._cnn_depth  
        depth = 2 ** i * self._cnn_depth
        self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
        self._conv_model.append(NormLayer(norm, depth))
        self._conv_model.append(self._act)
      self._conv_model = nn.Sequential(*self._conv_model)
    if len(self.mlp_keys) > 0:
      self._mlp_model = []
      for i, width in enumerate(self._mlp_layers):
        if i == 0:
          prev_width = np.prod(*[shapes[k] for k in self.mlp_keys]) 
        else:
          prev_width = self._mlp_layers[i-1]
        self._mlp_model.append(nn.Linear(prev_width, width))
        self._mlp_model.append(NormLayer(norm, width))
        self._mlp_model.append(self._act)
      self._mlp_model = nn.Sequential(*self._mlp_model)

  def forward(self, data):
    key, shape = list(self.shapes.items())[0]
    batch_dims = data[key].shape[:-len(shape)]
    data = {
        k: v.reshape((-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_keys:
      outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
    if self.mlp_keys:
      outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
    output = torch.cat(outputs, -1)
    return output.reshape(batch_dims + output.shape[1:])

  def _cnn(self, data):
    x = torch.cat(list(data.values()), -1)
    x = self._conv_model(x)
    return x.reshape(tuple(x.shape[:-3]) + (-1,))

  def _mlp(self, data):
    x = torch.cat(list(data.values()), -1)
    x = self._mlp_model(x)
    return x


class Decoder(nn.Module):
  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act=nn.ELU, norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400], embed_dim=1024):
    super().__init__()
    self._embed_dim = embed_dim
    self._shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Decoder CNN outputs:', list(self.cnn_keys))
    print('Decoder MLP outputs:', list(self.mlp_keys))
    self._act = act()
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers
    self.channels = {k: self._shapes[k][0] for k in self.cnn_keys}

    # Image-like
    if len(self.cnn_keys) > 0:
      self._conv_in = nn.Sequential(nn.Linear(embed_dim, 32*self._cnn_depth))
      self._conv_model = []
      for i, kernel in enumerate(self._cnn_kernels):
        if i == 0:
          prev_depth = 32*self._cnn_depth
        else:
          prev_depth = 2 ** (len(self._cnn_kernels) - (i - 1) - 2) * self._cnn_depth
        depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
        act, norm = self._act, self._norm
        if i == len(self._cnn_kernels) - 1:
          depth, act, norm = sum(self.channels.values()), nn.Identity(), 'none'
        self._conv_model.append(nn.ConvTranspose2d(prev_depth, depth, kernel, stride=2))
        self._conv_model.append(NormLayer(norm, depth))
        self._conv_model.append(act)
      self._conv_model = nn.Sequential(*self._conv_model)
    # Vector-like
    if len(self.mlp_keys) > 0:
      self._mlp_model = []
      for i, width in enumerate(self._mlp_layers):
        if i == 0:
          prev_width = embed_dim
        else:
          prev_width = self._mlp_layers[i-1]
        self._mlp_model.append(nn.Linear(prev_width, width))
        self._mlp_model.append(NormLayer(norm, width))
        self._mlp_model.append(self._act)
      self._mlp_model = nn.Sequential(*self._mlp_model)
      for key, shape in { k : shapes[k] for k in self.mlp_keys }.items():
        self.add_module(f'dense_{key}', DistLayer(width, shape))

  def forward(self, features):
    outputs = {}
    if self.cnn_keys:
      outputs.update(self._cnn(features))
    if self.mlp_keys:
      outputs.update(self._mlp(features))
    return outputs

  def _cnn(self, features):
    x = self._conv_in(features)
    x = x.reshape([-1, 32 * self._cnn_depth, 1, 1,])
    x = self._conv_model(x)
    x = x.reshape(list(features.shape[:-1]) + list(x.shape[1:])) 
    if len(x.shape) == 5:
      means = torch.split(x, list(self.channels.values()), 2) 
    else:
      means = torch.split(x, list(self.channels.values()), 1) 
    dists = {
        key: D.Independent(D.Normal(mean, 1), 3)
        for (key, shape), mean in zip(self.channels.items(), means)}
    return dists

  def _mlp(self, features):
    shapes = {k: self._shapes[k] for k in self.mlp_keys}
    x = features
    x = self._mlp_model(x)
    dists = {}
    for key, shape in shapes.items():
      dists[key] = getattr(self, f'dense_{key}')(x)
    return dists


class MLP(nn.Module):
  def __init__(self, in_shape, shape, layers, units, act=nn.ELU, norm='none', **out):
    super().__init__()
    self._in_shape = in_shape
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = act()
    self._out = out
    
    last_units = in_shape
    for index in range(self._layers):
      self.add_module(f'dense{index}', nn.Linear(last_units, units))
      self.add_module(f'norm{index}', NormLayer(norm, units))
      last_units = units
    self._out = DistLayer(units, shape, **out)

  def forward(self, features):
    x = features 
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = getattr(self, f'dense{index}')(x)
      x = getattr(self, f'norm{index}')(x)
      x = self._act(x)
    x = x.reshape(list(features.shape[:-1]) + [x.shape[-1]])
    return self._out(x)


class GRUCell(nn.Module):
  def __init__(self, inp_size, size, norm=False, act=nn.Tanh, update_bias=-1, device='cuda', **kwargs):
    super().__init__()
    self._inp_size = inp_size
    self._size = size
    self._act = act()
    self._norm = norm
    self._update_bias = update_bias
    self.device = device
    self._layer = nn.Linear(inp_size + size, 3 * size, bias=norm is not None, **kwargs)
    if norm:
      self._norm = nn.LayerNorm(3*size)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return torch.zeros((batch_size), self._size, device=self.device)

  @property
  def state_size(self):
    return self._size

  def forward(self, inputs, state):
    state = state[0]  # State is wrapped in a list.
    parts = self._layer(torch.cat([inputs, state], -1))
    if self._norm:
      parts = self._norm(parts)
    reset, cand, update = torch.chunk(parts, 3, -1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(nn.Module):
  def __init__(
      self, in_dim, shape, dist='mse', min_std=0.1, init_std=0.0, bias=True):
    super().__init__()
    self._in_dim = in_dim
    self._shape = shape if type(shape) in [list,tuple] else [shape]
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._out = nn.Linear(in_dim, int(np.prod(shape)) , bias=bias)
    if dist in ('normal', 'tanh_normal', 'trunc_normal'):
      self._std = nn.Linear(in_dim, int(np.prod(shape)) )

  def forward(self, inputs):
    out = self._out(inputs)
    out = out.reshape(list(inputs.shape[:-1]) + list(self._shape)) 
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self._std(inputs) 
      std = std.reshape(list(inputs.shape[:-1]) + list(self._shape)) 
    if self._dist == 'mse':
      dist = D.Normal(out, 1.0)
      return D.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = D.Normal(out, std)
      return D.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      out = torch.sigmoid(out)
      dist = BernoulliDist(out)
      return D.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * torch.tanh(out / 5)
      std = F.softplus(std + self._init_std) + self._min_std
      dist = utils.SquashedNormal(mean, std)
      dist = D.Independent(dist, len(self._shape))
      return SampleDist(dist)
    if self._dist == 'trunc_normal':
      mean = torch.tanh(out)
      std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = utils.TruncatedNormal(mean, std)
      return D.Independent(dist, 1)
    if self._dist == 'onehot':
      return OneHotDist(out)
    raise NotImplementedError(self._dist)


class NormLayer(nn.Module):
  def __init__(self, name, dim=None):
    super().__init__()
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      assert dim != None
      self._layer = nn.LayerNorm(dim)
    else:
      raise NotImplementedError(name)

  def forward(self, features):
    if self._layer is None:
      return features
    return self._layer(features)

class Optimizer:
  def __init__(
      self, name, parameters, lr, eps=1e-4, clip=None, wd=None,
      opt='adam', wd_pattern=r'.*', use_amp=False):
    assert 0 <= wd < 1
    assert not clip or 1 <= clip
    self._name = name
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern
    self._opt = {
        'adam': lambda: torch.optim.Adam(parameters, lr, eps=eps),
        'nadam': lambda: torch.optim.Nadam(parameters, lr, eps=eps),
        'adamax': lambda: torch.optim.Adamax(parameters, lr, eps=eps),
        'sgd': lambda: torch.optim.SGD(parameters, lr),
        'momentum': lambda: torch.optim.SGD(lr, momentum=0.9),
    }[opt]()
    self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    self._once = True

  def __call__(self, loss, params):
    params = list(params)
    assert len(loss.shape) == 0 or (len(loss.shape) == 1 and loss.shape[0] == 1), (self._name, loss.shape)
    metrics = {}

    # Count parameters.
    if self._once:
      count = sum(p.numel() for p in params if p.requires_grad) 
      print(f'Found {count} {self._name} parameters.')
      self._once = False

    # Check loss.
    metrics[f'{self._name}_loss'] = loss.detach().cpu().numpy()

    # Compute scaled gradient.
    self._scaler.scale(loss).backward()
    self._scaler.unscale_(self._opt)

    # Gradient clipping.
    if self._clip:
      norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
      metrics[f'{self._name}_grad_norm'] = norm.item()
  
    # Weight decay.
    if self._wd:
      self._apply_weight_decay(params)
    
    # Apply gradients.
    self._scaler.step(self._opt)
    self._scaler.update()
    
    self._opt.zero_grad() 
    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      raise NotImplementedError('Non trivial weight decay')
    else:
      for var in varibs:
        var.data = (1 - self._wd) * var.data

class StreamNorm:
  def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8, device='cuda'):
    # Momentum of 0 normalizes only based on the current batch.
    # Momentum of 1 disables normalization.
    self.device = device
    self._shape = tuple(shape)
    self._momentum = momentum
    self._scale = scale
    self._eps = eps
    self.mag = torch.ones(shape).to(self.device) 

  def __call__(self, inputs):
    metrics = {}
    self.update(inputs)
    metrics['mean'] = inputs.mean()
    metrics['std'] = inputs.std()
    outputs = self.transform(inputs)
    metrics['normed_mean'] = outputs.mean()
    metrics['normed_std'] = outputs.std()
    return outputs, metrics

  def reset(self):
    self.mag = torch.ones_like(self.mag).to(self.device)

  def update(self, inputs):
    batch = inputs.reshape((-1,) + self._shape)
    mag = torch.abs(batch).mean(0) 
    self.mag.data = self._momentum * self.mag.data + (1 - self._momentum) * mag

  def transform(self, inputs):
    values = inputs.reshape((-1,) + self._shape)
    values /= self.mag[None] + self._eps 
    values *= self._scale
    return values.reshape(inputs.shape)

class RequiresGrad:
  def __init__(self, model):
    self._model = model

  def __enter__(self):
    self._model.requires_grad_(requires_grad=True)

  def __exit__(self, *args):
    self._model.requires_grad_(requires_grad=False)
