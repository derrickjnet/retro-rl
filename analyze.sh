ls $1/*/log | grep Sonic | xargs -i bash -c "echo {}; tail -100000 {} | grep EPISODE | tail -1"
