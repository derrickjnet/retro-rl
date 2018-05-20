import numpy as np
from baselines.common.vec_env import VecEnv

class ExplorationVecEnv(VecEnv):
    def __init__(self, vecenv, exploration_f, state_encoder=None):
        VecEnv.__init__(self, vecenv.num_envs, vecenv.observation_space, vecenv.action_space)
        self.vecenv = vecenv
        self.state_encoder = state_encoder
        self.explorations = [ exploration_f(vecenv.env_ids[env_idx]) for env_idx in range(0, self.num_envs) ]
        self.actions = None   

    def action_metas(self, action_metas):
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          exploration.action_meta(action_metas[env_idx])
 
    def step_async(self, actions):
        assert self.actions is None
        self.actions = actions
        self.vecenv.step_async(actions)
 
    def step_wait(self):
        assert self.actions is not None
        obses, rews, dones, infos = self.vecenv.step_wait()
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
        obses = self.vecenv.reset()
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          exploration.reset(obses[env_idx])
        return obses

    def close(self):
        for env_idx in range(self.num_envs):
          self.explorations[env_idx].close()
        self.vecenv.close()

