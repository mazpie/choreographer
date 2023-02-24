import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import envs
import utils
from logger import Logger
from replay import ReplayBuffer, make_replay_loader

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_spec=action_spec)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.workdir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = cfg.task if cfg.task != 'none' else PRIMAL_TASKS[self.cfg.domain] # -> which is the URLB default
        frame_stack = 1
        img_size = 64

        self.train_env = envs.make(task, cfg.obs_type, frame_stack, 
                                  cfg.action_repeat, cfg.seed, img_size=img_size) 
        self.eval_env = envs.make(task, cfg.obs_type, frame_stack, 
                                 cfg.action_repeat, cfg.seed, img_size=img_size)

        # # create agent 
        self.agent = make_agent(self.train_env.obs_space,
                                self.train_env.action_spec(), cfg, cfg.agent)
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create replay storage
        self.replay_storage = ReplayBuffer(data_specs, meta_specs,
                                                  self.workdir / 'buffer',
                                                  length=cfg.batch_length, **cfg.replay,
                                                  device=cfg.device)

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.batch_size, # 
                                                )
        self._replay_iter = None

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            agent_state = None
            while not time_step['is_last']:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, agent_state = self.agent.act(time_step, # time_step.observation
                                            meta,
                                            self.global_step,
                                            eval_mode=True,
                                            state=agent_state)
                time_step = self.eval_env.step(action)
                total_reward += time_step['reward']
                step += 1

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        train_every_n_steps = self.cfg.train_every_actions // self.cfg.action_repeat 
        should_train_step = utils.Every(train_every_n_steps * self.cfg.action_repeat,  
                                      self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, 
                                      self.cfg.action_repeat)
        should_log_recon = utils.Every(self.cfg.recon_every_frames, 
                                        self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        agent_state = None
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta) 
        metrics = None
        while train_until_step(self.global_step):
            if time_step['is_last']:
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage)) 
                        log('step', self.global_step)
                    # save last model
                    self.save_last_model()

                # reset env
                time_step = self.train_env.reset()
                agent_state = None # Resetting agent's latent state
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta) 
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step > 0:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if  seed_until_step(self.global_step):
                    action =  self.train_env.act_space['action'].sample()
                else:
                    action, agent_state = self.agent.act(time_step, # time_step.observation
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            state=agent_state)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if should_train_step(self.global_step):
                    metrics = self.agent.update(next(self.replay_iter), self.global_step)[1] 
                if should_log_scalars(self.global_step):
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if self.global_step > 0 and should_log_recon(self.global_step):
                    videos = self.agent.report(next(self.replay_iter))
                    self.logger.log_video(videos, self.global_frame)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step['reward']
            self.replay_storage.add(time_step, meta) 
            episode_step += 1
            self._global_step += 1

    @utils.retry
    def save_snapshot(self):
        snapshot = self.get_snapshot_dir() / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = '_'.join([
            cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
            str(cfg.seed)
        ])
        wandb.init(project=cfg.project_name + "_pretrain", group=cfg.agent.name, name=exp_name)
        wandb.config.update(cfg)
        self.wandb_run_id = wandb.run.id

    @utils.retry
    def save_last_model(self):
        snapshot = self.root_dir / 'last_snapshot.pt'
        if snapshot.is_file():
            temp = Path(str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt"))
            os.replace(snapshot, temp)
        keys_to_save = ['agent', '_global_step', '_global_episode']
        if self.cfg.use_wandb: 
            keys_to_save.append('wandb_run_id')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        try:
            snapshot = self.root_dir / 'last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = self.root_dir / 'second_last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id':
                assert wandb.run is None
                cfg = self.cfg
                exp_name = '_'.join([
                    cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                    str(cfg.seed)
                ])
                wandb.init(project=cfg.project_name + "_pretrain", group=cfg.agent.name, name=exp_name, id=v, resume="must")

    def get_snapshot_dir(self):
        snap_dir = self.cfg.snapshot_dir
        snapshot_dir = self.workdir / Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        return snapshot_dir 

@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()   
    workspace.train()

if __name__ == '__main__':
    main()
