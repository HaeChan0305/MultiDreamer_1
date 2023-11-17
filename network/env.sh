# System config
export HOME=/root/MultiDreamer/ROCA2
export OMP_NUM_THREADS=1
export NUM_WORKERS=4
export SEED=2021

# NOTE: Change the data config based on your detup!
# JSON files
export DATA_DIR=$HOME/ROCA-Public/Data/Dataset
# Resized images with intrinsics and poses
export IMAGE_ROOT=$HOME/ROCA-Public/Data/Images
# Depths and instances rendered over images
export RENDERING_ROOT=$HOME/ROCA-Public/Data/Rendering
# Scan2CAD Full Annotations
export FULL_ANNOT=$HOME/ROCA-Public/Data/full_annotations.json

# Model configurations
export RETRIEVAL_MODE=resnet_resnet+image+comp
export E2E=1
export NOC_WEIGHTS=1

# Train and test behavior
export EVAL_ONLY=1
export CHECKPOINT=$HOME/ROCA-Public/Data/model_best.pth  # "none"
export RESUME=1  # RESUME=0 means from last checkpoint
export OUTPUT_DIR=$HOME/ROCA-Public/Experiments
