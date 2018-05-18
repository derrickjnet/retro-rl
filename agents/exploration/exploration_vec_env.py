class ExplorationVecEnv(VecEnv, exploration_f, state_encoder=None):
    def __init__(self, vecenv, state_encoder=None):
        VecEnv.__init__(self, vecenv.num_envs, vecenv.observation_space, vecenv.action_space)
        self.state_encoder = state_encoder
        self.explorations = [ exploration_f(self.envs[env_idx].spec, env_idx) for env_idx in range(0, self.num_envs) ]

    def step(self, actions):
        obses, rews, dones, infos = vecenv.step(actions)
        state_embeddings = state_encoder.encode_states(obses)
        final_rewards = []
        for (env_idx, exploration) in zip(self.explorations, range(0, self.num_envs)):
          final_reward = exploration.step(action, obses[env_idx], rews[env_idx], dones[env_idx], infos[env_idx])
          final_rewards.append(final_reward)
        return obses, final_rewards, dones, infos

    def reset(self):
        obses = vecenv.reset()
        for (env_idx, exploration) in zip(self.explorations, range(0, self.num_envs)):
          epxloration.reset(obses[env_idx])
        return obses

    def close(self):
        vecenv.close()
