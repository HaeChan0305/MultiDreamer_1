# source activate sam
# cd SAM
# BBOX=$(python inference_auto_generation.py --level 2 --input "18.png")
# conda deactivate
# conda activate sam2
# cd ../StableDiffusionInpaint
# python inpainting.py --input "18.png" --bbox "$BBOX"
# # 18, 36

input_img=(4)

source activate sam

cd SAM

# for ((i=61; i<=68; i++)); do
for i in "${input_img[@]}"; do
    INPUT_IMAGE="${i}.png"
    BBOX=$(python inference_auto_generation.py --level 2 --input "${INPUT_IMAGE}")
    
    conda deactivate
    conda activate sam2
    
    cd ../StableDiffusionInpaint
    
    python inpainting.py --input "${INPUT_IMAGE}" --bbox "${BBOX}"
    
    conda deactivate
    source activate sam
    cd ../SAM
done
