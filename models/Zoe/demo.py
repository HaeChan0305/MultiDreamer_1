"""
  conda activate sync-haechan
  python demo.py --input 1 --gpu 1
"""
import argparse

from torchvision.transforms import ToTensor
from PIL import Image
import torch
import numpy as np

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points

def main(input):
    # ****************** [1] prepare model ******************
    print("prepare zoe model")
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
    assert torch.cuda.is_available()
    DEVICE = "cuda"

    conf = get_config()
    model = build_model(conf).to(DEVICE)
    model.eval()

    # ****************** [2] prepare img ******************
    img = Image.open(f"../../data/input/eval/{input}.png")
    X = ToTensor()(img)

    if X.shape[0] == 4 : # if RGBA image transform to RGB format
        X = X[:3, :, :]

    X = X.unsqueeze(0).to(DEVICE)

    # ****************** [3-1] predict depth ******************

    print("start predictiong")
    with torch.no_grad():
        out = model.infer(X).cpu() #(1, H, W) : 1.xx ~ 2.xx

    # ****************** [3-2] depth_to_points ******************
    print("convert to points")
    pts3d = depth_to_points(out[0].numpy(), R=None, t=None)
    output_path = f"../../data/output/{input}/depth.npy"
    np.save(output_path, pts3d)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=int)
    parser.add_argument('--gpu', type=int)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu)
    print(f"Run ZoeDepth with GPU: {opt.gpu}")

    main(opt.input)

