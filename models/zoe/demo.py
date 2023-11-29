
"""
  Example/
    conda activate multidreamer_2
    python demo.py --indir "../../data/input" --input 21 --outidr "../../data/output" --gpu 1

  Option/
    --indir 
      : (Require, str) The path of the folder where input image exist
    --input 
      : (Required, int) The input image number, filename of the image should be {input}.png
    --outdir 
      : (Require, str) The path where the result will be store,
        save depth to points value as .npy file in the subdirectory, {input}/depth.npy
    --gpu 
      : (Optional, int) If you want to specify gpu number, using default setting when no option
"""

import argparse
import os

from torchvision.transforms import ToTensor
from PIL import Image
import torch
import numpy as np

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points

def main(arg):
    # [1] prepare model
    print("prepare zoe model")
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
    assert torch.cuda.is_available()
    DEVICE = "cuda"

    conf = get_config()
    model = build_model(conf).to(DEVICE)
    model.eval()

    # [2] prepare img
    img = Image.open(arg.indir + f"/{input}.png")
    X = ToTensor()(img)

    if X.shape[0] == 4 : # if RGBA image transform to RGB format
        X = X[:3, :, :]

    X = X.unsqueeze(0).to(DEVICE)

    # [3-1] predict depth
    with torch.no_grad():
        out = model.infer(X).cpu() #(1, H, W) : 1.xx ~ 2.xx

    # [3-2] depth_to_points
    pts3d = depth_to_points(out[0].numpy(), R=None, t=None)
    output_path = arg.outdir + f"/{input}/depth.npy"

    if not os.path.exists(arg.outdir + f"/{input}"):
        os.makedirs(arg.outdir + f"/{input}")
    np.save(output_path, pts3d)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=int)
    parser.add_argument("--indir", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument('--gpu', type=int)

    arg = parser.parse_args()

    if arg.gpu != None :
        torch.cuda.set_device(arg.gpu)

    main(arg)