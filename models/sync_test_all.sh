# source activate sync-haechan

cd /root/MultiDreamer/models/Sync # model/Sync

NUM=48

# get input from data/input/{number}.png
python generate.py --input $NUM --ground_truth --mesh --gpu 0 &
pid_gt=$!

# get input from data/output/{number}/inpainting{index}.png
python generate.py --input $NUM --index 0 --mesh --gpu 1 &
pid_obj1=$!
python generate.py --input $NUM --index 1 --mesh --gpu 2 &
pid_obj2=$!

wait $pid_gt
wait $pid_obj1
wait $pid_obj2

# conda deactivate

# make meshes at data/output/{number}/mesh{index}.ply