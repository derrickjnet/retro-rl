export RETRO_RECORD=${RETRO_PATH:-results/}${RETRO_AGENT}-${RETRO_CONF}/$RETRO_GAME-$RETRO_STATE
mkdir -p $RETRO_RECORD && \
mkdir $RETRO_RECORD/tensorflow && \
RETRO_LOGDIR=$RETRO_RECORD/tensorflow python ${RETRO_AGENT}_agent.py >> $RETRO_RECORD/main.log 2>&1
