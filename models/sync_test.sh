source activate sync-haechan
cd Sync

NUM=4

# get input from data/output/{number}/inpainting{index}.png
python generate.py --input $NUM --index 0 --mesh --gpu 1 &
pid_obj1=$!
python generate.py --input $NUM --index 1 --mesh --gpu 2 &
pid_obj2=$!

wait $pid_obj1
wait $pid_obj2

# get input from data/input/eval/{number}.png
python generate.py --input $NUM --ground_truth --mesh --gpu 1 &
pid_gt=$!

cd ../Zoe
python demo.py --input $NUM --gpu 2 &
pid_zoe=$!

wait $pid_gt
wait $pid_zoe

conda deactivate

# make meshes at data/output/{number}/mesh{index}.ply