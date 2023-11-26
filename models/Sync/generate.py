"""
    // ground_truth 돌릴 때 --ground_truth 옵션 넣기
    // mesh 까지 만들고 싶을 때 --mesh 옵션 넣기
    // gpu 지정하고 싶을 때 --gpu {gpu_num} 옵션 넣기, default는 1

    >> ground_truth 돌리고 싶을 때
    python generate.py --input 1 --index 0 --ground_truth --mesh

    >> inpainting 결과 돌리고 싶을 때, index는 inpainting결과 index
    python3 generate.py --input 1 --index 0 --mesh

"""

import argparse
from pathlib import Path
import time

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

def generate(name, index, ground_truth=False, mesh=False, gpu=1):
    cfg = 'configs/syncdreamer.yaml'
    ckpt = 'ckpt/syncdreamer-pretrain.ckpt'
    seed = 6033

    if ground_truth == True :
        input_path = "../../data/input/eval/" + name + ".png"
        image_path = "../../data/input/eval/sync_input" + name + ".png"
        output_path = "../../data/output/" + name + "/sync_output.png" 
    else :
        input_path = "../../data/output/" + name + "/inpainting" + index + ".png"
        image_path = "../../data/output/" + name + "/sync_input" + index + ".png"
        output_path = "../../data/output/" + name + "/sync_output" + index + ".png"
    
    print(input_path, image_path, output_path)
    process(input_path, image_path)

    torch.random.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(gpu)
    model = load_model(cfg, ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    # Path(f'{output_path}').mkdir(exist_ok=True, parents=True)

    print("-"*10, "start prediction", "-"*10)

    # prepare data
    data = prepare_inputs(image_path, 30, 200)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], 1, dim=0)


    sampler = SyncDDIMSampler(model, 50)

    x_sample = model.sample(sampler, data, 2.0, 8)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    for bi in range(B):
        imsave(output_path, np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))

    if mesh:
        render_mesh(name, index, ground_truth, gpu)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=int)
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--mesh', action='store_true')
    parser.add_argument('--ground_truth', action='store_true')
    parser.add_argument('--gpu', type=int)

    opt = parser.parse_args()
    print(" >> Run SyncDreamer with GPU ", opt.gpu)

    generate(str(opt.input), str(opt.index), opt.ground_truth, opt.mesh, opt.gpu)

