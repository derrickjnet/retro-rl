from baselines.common.vec_env import VecEnv

class RewardScalingVecEnv(VecEnv):
    def __init__(self, vecenv, reward_scale):
        VecEnv.__init__(self, vecenv.num_envs, vecenv.observation_space, vecenv.action_space)
        self.actions = None
        self.vecenv = vecenv
        self.reward_scale = reward_scale

    def action_metas(self, action_metas):
        self.vecenv.action_metas(action_metas)
    
    def step_async(self, actions):
        self.actions = actions
        self.vecenv.step_async(actions) 
 
    def step_wait(self):
        obses, rews, dones, infos = self.vecenv.step_wait()
        self.actions = None
        return obses, rews * self.reward_scale, dones, infos

    def reset(self):
        return self.vecenv.reset()

    def close(self):
        self.vecenv.close()

