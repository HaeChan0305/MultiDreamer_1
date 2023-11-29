
# 18, 21, 24, 29, 36, 37, 40, 42

#----------------------
cd Sync 

python generate.py --input 21 --ground_truth --mesh --gpu 1 &
pid_obj1=$!
python generate.py --input 29 --ground_truth --mesh --gpu 2 &
pid_obj2=$!

wait $pid_obj1
wait $pid_obj2

python generate.py --input 36 --ground_truth --mesh --gpu 1 &
pid_obj1=$!
python generate.py --input 37 --ground_truth --mesh --gpu 2 &
pid_obj2=$!

wait $pid_obj1
wait $pid_obj2

python generate.py --input 40 --ground_truth --mesh --gpu 1 &
pid_obj1=$!
python generate.py --input 42 --ground_truth --mesh --gpu 2 &
pid_obj2=$!

wait $pid_obj1
wait $pid_obj2

#----------------------

# make meshes at data/output/{number}/mesh{index}.ply