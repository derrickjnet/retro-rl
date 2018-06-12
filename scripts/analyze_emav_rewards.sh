bash scripts/analyze.sh $1 | grep emav_episode_reward | sed -e s"/^.*emav_episode_reward=//"
