source ~/anaconda3/etc/profile.d/conda.sh

# Main pipline

INPUT=21 # name of input image in the INPUT_DIR
INPUT_IMAGE="/root/MultiDreamer/data/input/1.png"
OUTPUT_DIR="/root/MultiDreamer/data/merging_test/1/"

# ---------- [1] Object Detachment ----------
conda activate sam
cd /root/MultiDreamer/models/SAM
BBOX=$(python inference_auto_generation.py --input "${INPUT_IMAGE}" --output_dir "${OUTPUT_DIR}" --level 2) 
conda deactivate

conda activate sam2
cd /root/MultiDreamer/models/StableDiffusionInpaint
python inpainting.py --input "${INPUT_IMAGE}" --output_dir "${OUTPUT_DIR}" --bbox "${BBOX}"
conda deactivate



# ---------- [2] Mesh Reconstruction ----------
# source activate sync-haechan
# cd models
# # conda activate syncdreamer_2

# cd sync

# python generate.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT --index 0 --mesh
# python generate.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT --index 1 --mesh

# # if you want to see the result of baseline together
# python generate.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT --baseline --mesh

# cd ..
# cd zoe

# python demo.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT

# cd ..

# conda deactivate
# cd ..

# ---------- [3] Mesh Alignment ----------