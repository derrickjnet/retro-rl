RETRO_CONF=$1
RETRO_COLLECT_START=$2
RETRO_COLLECT_STOP=$3
mkdir -p /data.archive/home/ubuntu/results/${RETRO_CONF}/events
ls -1 /data/home/ubuntu/results/${RETRO_CONF}/ | xargs -P 100 -i bash -c "python collect_policy_targets.py /data/home/ubuntu/results/${RETRO_CONF}/ {} $RETRO_COLLECT_START $RETRO_COLLECT_STOP /data.archive/home/ubuntu/results/${RETRO_CONF}/events &> /data.archive/home/ubuntu/results/${RETRO_CONF}/events/{}.log"
