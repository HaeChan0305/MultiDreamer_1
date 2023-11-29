# source activate sync-haechan

# gt 없음 - 21, 24, 29, 36, 37, 40, 42
# 
# 애매한 전부 안됨 - 46 49 45
my_set=(46 49) # 
for NUM in "${my_set[@]}"; do
  cd /root/MultiDreamer/models/Sync # model/Sync

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

  cd /root/MultiDreamer/models/Zoe # model/Zoe
  python demo.py --input $NUM --gpu 1 &
  pid_zoe=$!

  wait $pid_zoe
done

# conda deactivate

# make meshes at data/output/{number}/mesh{index}.ply