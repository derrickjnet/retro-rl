bash analyze.sh $1 | grep avg_episode_reward= | sed -e s"/^.*avg_episode_reward=//"
