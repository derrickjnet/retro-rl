import os
import numpy as np
from baselines.common.vec_env import VecEnv

class ExplorationVecEnv(VecEnv):
    def __init__(self, vec_env, exploration_f, state_encoder=None, root_dir = os.environ['RETRO_ROOTDIR'], record_dir=os.environ.get('RETRO_RECORDDIR'), save_states=os.environ.get('RETRO_SAVESTATE') == "true"):
        VecEnv.__init__(self, vec_env.num_envs, vec_env.observation_space, vec_env.action_space)
        self.env_ids = vec_env.env_ids
        self.vec_env = vec_env
        self.state_encoder = state_encoder
        self.log_files = [ open(root_dir + "/" + self.env_ids[env_idx] + "/log", "w") for env_idx in range(self.num_envs) ]
        if save_states:
          self.save_state_dirs = [ record_dir + "/" + self.env_ids[env_idx] for env_idx in range(self.num_envs) ]
        else:
          self.save_state_dirs = [ None for env_idx in range(self.num_envs) ]
        self.explorations = [ exploration_f(env_idx, self.env_ids[env_idx], log_file=self.log_files[env_idx], save_state_dir=self.save_state_dirs[env_idx]) for env_idx in range(self.num_envs) ]
        self.actions = None   

    def action_metas(self, action_metas):
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          exploration.action_meta(action_metas[env_idx])
 
    def step_async(self, actions):
        assert self.actions is None
        self.actions = actions
        self.vec_env.step_async(actions)
 
    def step_wait(self):
        assert self.actions is not None
        obses, rews, dones, infos = self.vec_env.step_wait()
        if self.state_encoder is not None:
          state_embeddings = self.state_encoder.encode(np.expand_dims(obses[:,:,:,-1],-1))
        else:
          state_embeddings = [ None for env_idx in range(0, self.num_envs) ]
        final_rewards = []
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          final_reward = exploration.step(self.actions[env_idx], obses[env_idx], rews[env_idx], dones[env_idx], infos[env_idx], state_embeddings[env_idx])
          final_rewards.append(final_reward)
          if dones[env_idx]:
            exploration.reset(obses[env_idx])
        self.actions = None
        return obses, np.asarray(final_rewards), dones, infos

    def reset(self):
        obses = self.vec_env.reset()
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          exploration.reset(obses[env_idx])
        return obses

    def close(self):
        for env_idx in range(self.num_envs):
          self.explorations[env_idx].close()
          self.log_files[env_idx].close()
        self.vec_env.close()

