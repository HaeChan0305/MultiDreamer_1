# # conda environment
# conda create -n multidreamer python=3.8
# conda activate multidreamer

# # Simple deps
# pip install -r requirements.txt

# # pytorch3d
# conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
# conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# conda install -c bottler nvidiacub
# # NOTE: Tested with 0.5 and 0.6
# conda install pytorch3d -c pytorch3d-nightly

# # detectron2
# python -m pip install detectron2==0.3 -f \
#   https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

# # numba
# conda install numba


# conda environment
conda create -n multidreamer python=3.8
conda activate multidreamer

# Simple deps
pip install -r requirements.txt

# pytorch3d
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
# NOTE: Tested with 0.5 and 0.6
conda install pytorch3d -c pytorch3d-nightly

# detectron2
python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'

# numba
conda install numba
