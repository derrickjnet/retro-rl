ls $1/*/log | xargs -i bash -c "echo {}; tail -100000 {} | grep EPISODE | tail -1"
