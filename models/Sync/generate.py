"""
    testset 폴더에 {filename}.png 존재해야 함
    output 폴더에 {filename}.png로 나옴

    python3 generate.py --name sofatest
"""

"""
    python3 generate.py --ckpt ckpt/syncdreamer-pretrain.ckpt \
                    --input testset/sofatest.png \
                    --output output/sofatest \
                    --sample_num 4 \
                    --cfg_scale 2.0 \
                    --elevation 30 \
                    --crop_size 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs
from train_renderer import render_mesh

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/syncdreamer-pretrain.ckpt')
    parser.add_argument('--name',type=str, required=True)

    # parser.add_argument('--output', type=str, required=True)
    # parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--elevation', type=float, default=30)

    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=200)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    flags = parser.parse_args()

    input_path = "testset/" + flags.name + ".png"
    output_path = "output"

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{output_path}').mkdir(exist_ok=True, parents=True)

    print("-"*10, "start prediction", "-"*10)

    # prepare data
    data = prepare_inputs(input_path, flags.elevation, flags.crop_size)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)

    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    x_sample = model.sample(sampler, data, flags.cfg_scale, flags.batch_view_num)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    for bi in range(B):
        output_fn = Path(output_path)/ f'{flags.name}.png'
        imsave(output_fn, np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))

    render_mesh(flags.name)

if __name__=="__main__":
    main()

