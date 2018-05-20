from baselines.common.vec_env import VecEnv

class ExplorationVecEnv(VecEnv):
    def __init__(self, vecenv, exploration_f, state_encoder=None):
        VecEnv.__init__(self, vecenv.num_envs, vecenv.observation_space, vecenv.action_space)
        self.vecenv = vecenv
        self.state_encoder = state_encoder
        self.env_ids = [self.envs[env_idx].spec.id for env_idx in range(0, self.num_envs) ]
        assert len(set(self.env_ids)) == len(self.env_ids)
        self.explorations = [ exploration_f(self.env_ids[env_idx]) for env_idx in range(0, self.num_envs) ]

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def step_async(self, actions):
        self.actions = actions
 
    def step_wait(self):
        obses, rews, dones, infos = vecenv.step(actions)
        if state_encoder is not None:
          state_embeddings = state_encoder.encode_states(obses)
        else:
          state_embeddings = [ None for env_idx in range(0, self.num_envs) ]
        final_rewards = []
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          final_reward = exploration.step(action, obses[env_idx], rews[env_idx], dones[env_idx], infos[env_idx], state_embeddings[env_idx])
          final_rewards.append(final_reward)
        return obses, final_rewards, dones, infos

    def reset(self):
        obses = vecenv.reset()
        for (exploration, env_idx) in zip(self.explorations, range(0, self.num_envs)):
          epxloration.reset(obses[env_idx])
        return obses

    def close(self):
        vecenv.close()
