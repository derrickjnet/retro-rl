ls $1/*/log | grep Sonic | xargs -i bash -c "echo {}; tail -n 100000 {} | grep total_rew | tail -n 1"

