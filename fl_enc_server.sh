#!/bin/bash
#python3 deleteFL_Enc.py
echo "Starting key manager"
echo "Starting server"
python3 key_manager.py &

python3 server.py &
# sleep 50

# range=$(python3 -c "from config_FL import num_client; print(num_client())")
# new_range=$((range-1))

# for i in $(seq 0 $new_range); do
#     #sleep 1
#     echo "Starting client $i"
#     python3 FL_client.py --partition=${i} &
# done

# Wait for all background processes to complete
read