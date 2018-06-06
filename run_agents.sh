for RETRO_GAMESTATE in $(cat $1 | grep -v game | sed -e 's/\r//'); do
  export RETRO_GAME=`echo $RETRO_GAMESTATE | cut -d , -f 1`
  export RETRO_STATE=`echo $RETRO_GAMESTATE | cut -d , -f 2`
  echo Starting $RETRO_GAME-$RETRO_STATE
  nohup bash run_agent.sh&
done
