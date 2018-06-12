bash analyze.sh $1 | grep total_reward | sed -e "s/.*total_reward=//" | sed -e "s/ .*//"
