import os
import numpy as np
from anyrl.envs.wrappers.batched import BatchedWrapper 

class ExplorationBatchedEnv(BatchedWrapper):
    def __init__(self, batched_env, exploration_f, state_encoder=None, record_root_dir=os.environ['RETRO_RECORD']):
        BatchedWrapper.__init__(self, batched_env)
        self.env_ids = batched_env.env_ids
        self.batched_env = batched_env
        self.state_encoder = state_encoder
        self.record_dirs = [ record_root_dir + "/" + self.env_ids[env_idx] for env_idx in range(self.num_envs) ]
        self.log_files = [ open(self.record_dirs[env_idx] + "/log", "w") for env_idx in range(self.num_envs) ]
        self.explorations = [ exploration_f(self.env_ids[env_idx], log_file=self.log_files[env_idx], save_state_dir=self.record_dirs[env_idx]) for env_idx in range(self.num_envs) ]
        self.actions = {}

    def action_metas(self, action_metas, sub_batch=0):
        env_base_idx = sub_batch*self.batched_env.num_envs_per_sub_batch
        for env_offset_idx in range(0, self.batched_env.num_envs_per_sub_batch):
          self.explorations[env_base_idx + env_offset_idx].action_meta(action_metas[env_offset_idx])
 
    def step_start(self, actions, sub_batch=0):
        assert not sub_batch in self.actions
        self.actions[sub_batch] = actions
        self.batched_env.step_start(actions, sub_batch=sub_batch)
 
    def step_wait(self, sub_batch=0):
        assert sub_batch in self.actions
        obses, rews, dones, infos = self.batched_env.step_wait(sub_batch=sub_batch)
        if self.state_encoder is not None:
          state_embeddings = self.state_encoder.encode(np.expand_dims(np.stack(obses, axis=0)[:,:,:,-1],-1))
        else:
          state_embeddings = [ None for env_idx in range(0, self.num_envs) ]
        final_rewards = []
        env_base_idx = sub_batch*self.batched_env.num_envs_per_sub_batch
        for env_offset_idx in range(0, self.batched_env.num_envs_per_sub_batch):
          final_reward = self.explorations[env_base_idx + env_offset_idx].step(self.actions[sub_batch][env_offset_idx], obses[env_offset_idx], rews[env_offset_idx], dones[env_offset_idx], infos[env_offset_idx], state_embeddings[env_offset_idx])
          final_rewards.append(final_reward)
        del self.actions[sub_batch]
        return obses, np.asarray(final_rewards), dones, infos
 
    def reset_start(self, sub_batch=0):
        self.batched_env.reset_start(sub_batch=sub_batch)

    def reset_wait(self, sub_batch=0):
        obses = self.batched_env.reset_wait(sub_batch=sub_batch)
        env_base_idx = sub_batch*self.batched_env.num_envs_per_sub_batch
        for env_offset_idx in range(0, self.batched_env.num_envs_per_sub_batch):
          self.explorations[env_offset_idx + env_base_idx].reset(obses[env_offset_idx])
        return obses

    def close(self):
        for env_idx in range(self.num_envs):
          self.explorations[env_idx].close()
          self.log_files[env_idx].close()
        self.batched_env.close()

