import collections
import datetime
import io
import pathlib
import uuid

import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader
import torch
import utils

class ReplayBuffer(IterableDataset):

  def __init__(
      self, data_specs, meta_specs, directory, length=20, capacity=0, ongoing=False, minlen=1, maxlen=0,
      prioritize_ends=False, device='cuda', load_first=False):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    self._capacity = capacity
    self._ongoing = ongoing
    self._minlen = minlen
    self._maxlen = maxlen
    self._prioritize_ends = prioritize_ends
    self._random = np.random.RandomState()
    # filename -> key -> value_sequence
    self._complete_eps = load_episodes(self._directory, capacity, minlen, load_first=load_first)
    # worker -> key -> value_sequence
    self._ongoing_eps = collections.defaultdict(
        lambda: collections.defaultdict(list))
    self._total_episodes, self._total_steps = count_episodes(directory)
    self._loaded_episodes = len(self._complete_eps)
    self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())
    self._length = length
    self._data_specs = data_specs
    self._meta_specs = meta_specs    
    self.device = device
    try:
      assert self._minlen <= self._length <= self._maxlen
    except:
      print("Incosistency between min/max/length in the replay buffer. Defaulting to (length): ", length)
      self._minlen = self._maxlen = self._length = length

  def __len__(self):
    return self._total_steps

  @property
  def stats(self):
    return {
        'total_steps': self._total_steps,
        'total_episodes': self._total_episodes,
        'loaded_steps': self._loaded_steps,
        'loaded_episodes': self._loaded_episodes,
    }

  def add(self, time_step, meta, worker=0):
    episode = self._ongoing_eps[worker]
    for spec in self._data_specs:
        value = time_step[spec.name]
        if np.isscalar(value):
            value = np.full(spec.shape, value, spec.dtype)
        assert spec.shape == value.shape and spec.dtype == value.dtype
        episode[spec.name].append(value)
    for spec in self._meta_specs:
        value = meta[spec.name]
        if np.isscalar(value):
            value = np.full(spec.shape, value, spec.dtype)
        assert spec.shape == value.shape and spec.dtype == value.dtype
        episode[spec.name].append(value)
    if type(time_step) == dict:
      if time_step['is_last']: 
        self.add_episode(episode)
        episode.clear()
    else:
      if time_step.last(): 
        self.add_episode(episode)
        episode.clear()

  def add_episode(self, episode):
    length = eplen(episode)
    if length < self._minlen:
      print(f'Skipping short episode of length {length}.')
      return
    self._total_steps += length
    self._loaded_steps += length
    self._total_episodes += 1
    self._loaded_episodes += 1
    episode = {key: convert(value) for key, value in episode.items()}
    filename = save_episode(self._directory, episode)
    self._complete_eps[str(filename)] = episode
    self._enforce_limit()

  def __iter__(self):
    sequence = self._sample_sequence()
    while True:
      chunk = collections.defaultdict(list)
      added = 0
      while added < self._length:
        needed = self._length - added
        adding = {k: v[:needed] for k, v in sequence.items()}
        sequence = {k: v[needed:] for k, v in sequence.items()}
        for key, value in adding.items():
          chunk[key].append(value)
        added += len(adding['action'])
        if len(sequence['action']) < 1:
          sequence = self._sample_sequence()
      chunk = {k: np.concatenate(v) for k, v in chunk.items()}
      chunk['is_terminal'] = chunk['discount'] == 0
      chunk = {k : torch.as_tensor(np.copy(v), device=self.device) for k, v in chunk.items()}
      yield chunk

  def _sample_sequence(self):
    episodes = list(self._complete_eps.values())
    if self._ongoing:
      episodes += [
          x for x in self._ongoing_eps.values()
          if eplen(x) >= self._minlen]
    episode = self._random.choice(episodes)
    total = len(episode['action'])
    length = total
    if self._maxlen:
      length = min(length, self._maxlen)
    # Randomize length to avoid all chunks ending at the same time in case the
    # episodes are all of the same length.
    length -= np.random.randint(self._minlen)
    length = max(self._minlen, length)
    upper = total - length + 1
    if self._prioritize_ends:
      upper += self._minlen
    index = min(self._random.randint(upper), total - length)
    sequence = {
        k: convert(v[index: index + length])
        for k, v in episode.items() if not k.startswith('log_')}
    sequence['is_first'] = np.zeros(len(sequence['action']), np.bool)
    sequence['is_first'][0] = True
    if self._maxlen:
      assert self._minlen <= len(sequence['action']) <= self._maxlen
    return sequence

  def _enforce_limit(self):
    if not self._capacity:
      return
    while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
      # Relying on Python preserving the insertion order of dicts.
      oldest, episode = next(iter(self._complete_eps.items()))
      self._loaded_steps -= eplen(episode)
      self._loaded_episodes -= 1
      del self._complete_eps[oldest]


def count_episodes(directory):
  filenames = list(directory.glob('*.npz'))
  num_episodes = len(filenames)
  if len(filenames) > 0 and "-" in str(filenames[0]):
    num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
  else:
    num_steps = sum(int(str(n).split('_')[-1][:-4]) - 1 for n in filenames)
  return num_episodes, num_steps

@utils.retry
def save_episode(directory, episode):
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  identifier = str(uuid.uuid4().hex)
  length = eplen(episode)
  filename = directory / f'{timestamp}-{identifier}-{length}.npz'
  with io.BytesIO() as f1:
    np.savez_compressed(f1, **episode)
    f1.seek(0)
    with filename.open('wb') as f2:
      f2.write(f1.read())
  return filename


def load_episodes(directory, capacity=None, minlen=1, load_first=False):
  # The returned directory from filenames to episodes is guaranteed to be in
  # temporally sorted order.
  filenames = sorted(directory.glob('*.npz'))
  if capacity:
    num_steps = 0
    num_episodes = 0
    ordered_filenames = filenames if load_first else reversed(filenames)
    for filename in ordered_filenames:
      if "-" in str(filename):
        length = int(str(filename).split('-')[-1][:-4])
      else:
        length = int(str(filename).split('_')[-1][:-4])
      num_steps += length
      num_episodes += 1
      if num_steps >= capacity:
        break
    if load_first:
      filenames = filenames[:num_episodes]
    else:
      filenames = filenames[-num_episodes:]
  episodes = {}
  for filename in filenames:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode {str(filename)}: {e}')
      continue
    episodes[str(filename)] = episode
  return episodes


def convert(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value


def eplen(episode):
  return len(episode['action']) - 1


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def make_replay_loader(buffer, batch_size):
    return DataLoader(buffer,
                        batch_size=batch_size,
                        drop_last=True,
                      )
