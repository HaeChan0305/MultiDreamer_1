source activate sam
cd SAM
BBOX=$(python inference_auto_generation.py --level 1 --input "ref_washing.png")
conda deactivate
conda activate sam2
cd ../StableDiffusionInpaint
python inpainting.py --input "ref_washing.png" --bbox "$BBOX"