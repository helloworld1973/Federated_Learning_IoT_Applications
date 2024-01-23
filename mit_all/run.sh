for i in `seq 0 45`; do
    echo "Starting client $i"
    python client$i.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
sleep 300000