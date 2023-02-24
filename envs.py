from collections import OrderedDict, deque
from typing import Any, NamedTuple
import os

import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import custom_dmc_tasks as cdmc
import gym
import pickle

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((np.int(np.prod(spec.shape))
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class DMC:
  def __init__(self, env):
    self._env = env 
    self._ignored_keys = []

  @property
  def obs_space(self):
    spaces = {
        'observation': self._env.observation_spec(), 
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box((spec.minimum)*spec.shape[0], (spec.maximum)*spec.shape[0], shape=spec.shape, dtype=np.float32)
    return {'action': action}

  def step(self, action):
    time_step = self._env.step(action)
    assert time_step.discount in (0, 1)
    obs = {
        'reward': time_step.reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'observation': time_step.observation,
        'action' : action,
        'discount': time_step.discount
    }
    return obs 

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'observation': time_step.observation,
        'action' : np.zeros_like(self.act_space['action'].sample()),
        'discount': time_step.discount
    }
    return obs

  def __getattr__(self, name):
    if name == 'obs_space':
        return self.obs_space
    if name == 'act_space':
        return self.act_space
    return getattr(self._env, name)


class SparseMetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
    ):
        import metaworld

        os.environ["MUJOCO_GL"] = "egl"

        # Construct the benchmark, sampling tasks
        self.ml1 = metaworld.ML1(f'{name}-v2', seed=seed) 

        # Create an environment with task `pick_place`
        env_cls = self.ml1.train_classes[f'{name}-v2']  
        self._env = env_cls()
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self._seed = seed
        self._tasks = self.ml1.test_tasks
        if name == 'reach':
            with open(f'../../../mw_tasks/reach_harder/{seed}.pickle', 'rb') as handle:
                self._tasks = pickle.load(handle)

    def observation_spec(self,):
        v = self.obs_space['observation']
        return specs.BoundedArray(name='observation', shape=v.shape, dtype=v.dtype, minimum=v.low, maximum=v.high)

    def action_spec(self,):
        return specs.BoundedArray(name='action',
            shape=self._env.action_space.shape, dtype=self._env.action_space.dtype, minimum=self._env.action_space.low, maximum=self._env.action_space.high)

    @property
    def obs_space(self):
        spaces = {
            "observation": gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            success += float(info["success"])
            reward += float(info["success"])
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "observation": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy(),
            "state": state,
            'action' : action,
            "success": success,
            'discount' : 1
        }
        return obs

    def reset(self):
        # Set task to ML1 choices
        task_id = np.random.randint(0,len(self._tasks))
        return self.reset_with_task_id(task_id)

    def reset_with_task_id(self, task_id):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
            
        # Set task to ML1 choices
        task = self._tasks[task_id]
        self._env.set_task(task)

        state = self._env.reset()
        # This ensures the first observation is correct in the renderer
        self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera)
        for site in self._env._target_site_config:
            self._env._set_pos_site(*site)
        self._env.sim._render_context_offscreen._set_mujoco_buffers()

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy(),
            "state": state,
            'action' : np.zeros_like(self.act_space['action'].sample()),
            "success": False,
            'discount' : 1
        }
        return obs

    def __getattr__(self, name):
        if name == 'obs_space':
            return self.obs_space
        if name == 'act_space':
            return self.act_space
        return getattr(self._env, name)

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()

  def reset_with_task_id(self, task_id):
    self._step = 0
    return self._env.reset_with_task_id(task_id)

def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, exorl=False):
    env = cdmc.make_jaco(task, obs_type, seed, img_size, exorl=exorl)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    env._size = (img_size, img_size)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, exorl=False):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        env._size = (img_size, img_size)
        env._camera = camera_id
    return env


def make(name, obs_type, frame_stack, action_repeat, seed, img_size=84, exorl=False):
    assert obs_type in ['states', 'pixels']
    domain, task = name.split('_', 1)
    if domain == 'mw':
        return TimeLimit(SparseMetaWorld(task, seed=seed, action_repeat=action_repeat, size=(img_size,img_size), camera='corner2'), 250)
    else:
        domain = dict(cup='ball_in_cup', point='point_mass').get(domain, domain)

        make_fn = _make_jaco if domain == 'jaco' else _make_dmc
        env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed, img_size, exorl=exorl)

        if obs_type == 'pixels':
            env = FrameStackWrapper(env, frame_stack)
        else:
            env = ObservationDTypeWrapper(env, np.float32)

        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
        env = ExtendedTimeStepWrapper(env)

        return DMC(env)
