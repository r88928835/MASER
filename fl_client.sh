range=$(python3 -c "from config_FL import num_client; print(num_client())")
new_range=$((range-1))

for i in $(seq 0 $new_range); do
   echo "Starting client $i"
   python3 client.py --partition=${i} &
done

# Wait for all background processes to complete
read