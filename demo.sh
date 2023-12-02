# Main pipline

INPUT=21 # name of input image in the INPUT_DIR
INPUT_DIR="/root/MultiDreamer/data/input"
OUTPUT_DIR="/root/MultiDreamer/data/output"

# ---------- [1] Object Detachment ----------



# ---------- [2] Mesh Reconstruction ----------
source activate sync-haechan
cd models
# conda activate syncdreamer_2

cd sync

python generate.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT --index 0 --mesh
python generate.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT --index 1 --mesh

# if you want to see the result of baseline together
python generate.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT --baseline --mesh

cd ..
cd zoe

python demo.py --indir $INPUT_DIR --outdir $OUTPUT_DIR --input $INPUT

cd ..

conda deactivate
cd ..

# ---------- [3] Mesh Alignment ----------