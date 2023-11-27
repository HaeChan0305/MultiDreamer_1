source activate sam
cd SAM
BBOX=$(python inference_auto_generation.py --level 2 --input "36.png")
conda deactivate
conda activate sam2
cd ../StableDiffusionInpaint
python inpainting.py --input "36.png" --bbox "$BBOX"


# source activate sam

# cd SAM

# for ((i=35; i<=40; i++)); do
#     INPUT_IMAGE="${i}.png"
#     BBOX=$(python inference_auto_generation.py --level 2 --input "${INPUT_IMAGE}")
    
#     conda deactivate
#     conda activate sam2
    
#     cd ../StableDiffusionInpaint
    
#     python inpainting.py --input "${INPUT_IMAGE}" --bbox "${BBOX}"
    
#     conda deactivate
#     source activate sam
#     cd ../SAM
# done
