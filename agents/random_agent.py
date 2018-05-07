import gym_remote.exceptions as gre
import gym_remote.client as grc

def main():
    print('connecting to remote environment')
    env = grc.RemoteEnv('tmp/sock')
    print('starting episode')
    env.reset()

    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(rew, done, info)
        #env.render()
        if done:
            print('episode complete')
            obs = env.reset()

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        print('exception', e)
