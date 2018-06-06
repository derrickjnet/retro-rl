import time
from anyrl.rollouts import Player
from anyrl.rollouts.rollers import _reduce_states, _inject_state, _reduce_model_outs

class BatchedPlayer(Player):
    """
    A Player that uses a BatchedEnv to gather transitions.
    """

    def __init__(self, batched_env, model, num_timesteps=1):
        self.batched_env = batched_env
        self.model = model
        self.num_timesteps = num_timesteps
        self._cur_states = None
        self._last_obses = None
        self._episode_ids = [list(range(start, start+batched_env.num_envs_per_sub_batch))
                             for start in range(0, batched_env.num_envs,
                                                batched_env.num_envs_per_sub_batch)]
        self._episode_steps = [[0] * batched_env.num_envs_per_sub_batch
                               for _ in range(batched_env.num_sub_batches)]
        self._next_episode_id = batched_env.num_envs
        self._total_rewards = [[0.0] * batched_env.num_envs_per_sub_batch
                               for _ in range(batched_env.num_sub_batches)]

    def play(self):
        if self._cur_states is None:
            self._setup()
        results = []
        for _ in range(self.num_timesteps):
            for i in range(self.batched_env.num_sub_batches):
                results.extend(self._step_sub_batch(i))
        return results

    def _step_sub_batch(self, sub_batch):
        model_outs = self.model.step(self._last_obses[sub_batch], self._cur_states[sub_batch])
        #BEGIN: action_metas
        self.batched_env.action_metas(model_outs['action_metas'], sub_batch=sub_batch)
        #END: action_metas
        self.batched_env.step_start(model_outs['actions'], sub_batch=sub_batch)
        outs = self.batched_env.step_wait(sub_batch=sub_batch)
        end_time = time.time()
        transitions = []
        for i, (obs, rew, done, info) in enumerate(zip(*outs)):
            self._total_rewards[sub_batch][i] += rew
            transitions.append({
                'obs': self._last_obses[sub_batch][i],
                'model_outs': _reduce_model_outs(model_outs, i),
                'rewards': [rew],
                'new_obs': (obs if not done else None),
                'info': info,
                'start_state': _reduce_states(self._cur_states[sub_batch], i),
                'episode_id': self._episode_ids[sub_batch][i],
                'episode_step': self._episode_steps[sub_batch][i],
                'end_time': end_time,
                'is_last': done,
                'total_reward': self._total_rewards[sub_batch][i]
            })
            if done:
                _inject_state(model_outs['states'], self.model.start_state(1), i)
                self._episode_ids[sub_batch][i] = self._next_episode_id
                self._next_episode_id += 1
                self._episode_steps[sub_batch][i] = 0
                self._total_rewards[sub_batch][i] = 0.0
            else:
                self._episode_steps[sub_batch][i] += 1
        self._cur_states[sub_batch] = model_outs['states']
        self._last_obses[sub_batch] = outs[0]
        return transitions

    def _setup(self):
        self._cur_states = []
        self._last_obses = []
        for i in range(self.batched_env.num_sub_batches):
            self._cur_states.append(self.model.start_state(self.batched_env.num_envs_per_sub_batch))
            self.batched_env.reset_start(sub_batch=i)
            self._last_obses.append(self.batched_env.reset_wait(sub_batch=i))
