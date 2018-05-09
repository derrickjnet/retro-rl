import numpy as np
import random

class RandomMoveExpert:
  def reset(self, num_actions, batch_size):
    self.num_actions = num_actions
    self.batch_size = batch_size
    self.states = [[0,0] for _ in range(0, batch_size)]

  def step(self, observations):
    result = []
    for env_idx in range(0,len(observations)):
      [move_steps, jump_steps] = self.states[env_idx]
      action_probs = np.asarray([0.0 for _ in range(0, self.num_actions)])
      while move_steps == 0:
        move_steps = int(1000 * (random.random() - 0.5))
        jump_steps = 0
        print("MOVE: env=%s move_steps=%s" % (env_idx, move_steps))
      if jump_steps == 0 and random.random() > 0.90:
        jump_steps = random.choice(range(0,10))
        print("JUMP: env=%s move_steps=%s jump_steps=%s" % (env_idx, move_steps, jump_steps))
      if jump_steps > 0:
        jump_steps -= 1
        action_probs[6] = 1.0
      elif move_steps > 0:
        move_steps -= 1
        action_probs[1] = 1.0
      elif move_steps < 0:
        move_steps += 1
        action_probs[0] = 1.0
      result.append(action_probs)
      self.states[env_idx] = [move_steps, jump_steps]
    return result
