ls /data/home/ubuntu/results/$1/*/log | sed 's/\/data//' | sed -e "s/log//" | xargs -i bash -c 'echo /data{}; mkdir -p /data.archive{}; rsync --size-only /data{}log /data.archive{}'
