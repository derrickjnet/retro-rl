import gym_remote.exceptions as gre
import gym_remote.client as grc
from sonic_util import make_env
def main():
    print('connecting to remote environment')
    env = make_env(stack=False)
    print('starting episode')
    env.reset()
    episode_step = 0
    episode_reward = 0

    while True:
        episode_step += 1
        #action = env.action_space.sample()
        # HilltopZone.Act1
        if episode_step < 52:
          action = 1
        elif episode_step < 63:
          action = 0
        elif episode_step < 85:
          action = episode_step % 2 
        elif episode_step < 95:
          action = 1
        elif episode_step < 155:
          action = 1
        elif episode_step < 160:
          action = 5
        else:
          if episode_step % 2 == 0:
            action = 1
          else:
            action = 5
        obs, rew, done, info = env.step(action)
        episode_reward += rew
        print(action)
        print(rew, done, info)
        print(episode_reward)
        env.render()
        if done:
            print('episode complete')
            obs = env.reset()
            episode_step = 0
            episode_reward = 0

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        print('exception', e)
