
"""
  Example/
    conda activate multidreamer_2
    python generate.py --indir "../../data/input" --outidr "../../data/output"
                        --input 21 --index 0
                        --mesh --gpu 1

  Option/
    --indir 
      : (Require, str) The path of the folder where input image exist
    --input 
      : (Required, int) The input image number, filename of the image should be {input}.png
    --outdir 
      : (Require, str) The path where the result will be store,
        save depth to points value as .npy file in the subdirectory, {input}/depth.npy

    --index 
      : (Optional, int) The index of an object using as a input image, {outdir}/{input}/inpainting{index}.png
    --mesh 
      : (Optional) Generate .ply mesh based on SyncDreamer output by training NeuS
    --baseline 
      : (Optional) If you want to run a baseline method
    --gpu 
      : (Optional, int) If you want to specify gpu number, using default setting when no option
"""

import os
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs
from train_renderer import render_mesh
from foreground_segment import process

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def generate(arg):
    # default settings
    cfg = 'configs/syncdreamer.yaml'
    ckpt = 'ckpt/syncdreamer-pretrain.ckpt'
    seed = 6033

    # path settings
    if not os.path.exists(arg.outdir + f"/{input}"):
        os.makedirs(arg.outdir + f"/{input}")

    if arg.baseline == True :
        input_path = arg.indir + f"/{arg.input}.png"
        image_path = arg.indir + f"/sync_input{arg.input}.png"
        output_path = arg.outdir + f"/{arg.input}/sync_output.png"
    else :
        input_path = arg.outdir + f"/{arg.input}/inpainting{arg.index}.png"
        image_path = arg.outdir + f"/{arg.input}/sync_input{arg.index}.png"
        output_path = arg.outdir + f"/{arg.input}/sync_output{arg.index}.png"
    
    # [1] remove background of input image (optional)
    process(input_path, image_path)

    torch.random.manual_seed(seed)
    np.random.seed(seed)

    # [2] prepare model and data
    model = load_model(cfg, ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)

    data = prepare_inputs(image_path, 30, 200)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], 1, dim=0)

    sampler = SyncDDIMSampler(model, 50)

    # [3] predict multi view images
    x_sample = model.sample(sampler, data, 2.0, 8)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    # [4] store output as png file
    for bi in range(B):
        imsave(output_path, np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))

    # [5] generate mesh by training NeuS
    if arg.mesh:
        render_mesh(arg)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=int)
    parser.add_argument("--indir", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)

    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--mesh', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--gpu', type=int)

    arg = parser.parse_args()

    if arg.gpu != None :
        torch.cuda.set_device(arg.gpu)

    generate(arg)

