source activate sync-haechan
cd Sync

# python generate.py --input 1 --ground_truth --mesh --gpu 0 &
# pid_gt=$!

python generate.py --input 1 --index 0 --mesh --gpu 1 &
pid_obj1=$!

python generate.py --input 1 --index 1 --mesh --gpu 2 &
pid_obj2=$!

# wait $pid_gt
wait $pid_obj1
wait $pid_obj2

conda deactivate