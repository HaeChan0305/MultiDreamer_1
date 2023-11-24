import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file

from tasks import interactive_infer_image_idino_m2m_auto

def parse_option():
    parser = argparse.ArgumentParser('SemanticSAM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/semantic_sam_only_sa-1b_swinL.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--ckpt', default="models/swinl_only_sam_many2many.pth", metavar="FILE", help='path to ckpt', )
    parser.add_argument('--level', default=1, )
    parser.add_argument('--input', default="images/sofa_table.jpg", help='path to the input image', )
    parser.add_argument('--output', default="outputs/", help='path to the output image', )
    args = parser.parse_args()
    return args

args = parse_option()
cur_model = 'None'

model=None
model_size=None
ckpt=None
cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
      'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

sam_cfg=cfgs['L']
opt = load_opt_from_config_file(sam_cfg)
model_sam = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()

@torch.no_grad()
def inference(image,level=1,*args, **kwargs):
    level = [level]
    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh='','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False
        model=model_sam
        a = interactive_infer_image_idino_m2m_auto(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, *args, **kwargs)
        return a

input_image = Image.open(args.input).convert("RGB")
draw = ImageDraw.Draw(input_image)
results = inference(input_image, args.level)

new_results = []
for i, result in enumerate(results):
    seg = result['segmentation']
    if seg[0][0]:
        continue
    
    else:
        seg = seg.astype(np.uint8) * 255
        segmentation = Image.fromarray(seg)
        segmentation = segmentation.resize(input_image.size)
        seg = np.array(segmentation)
        segmentation.save(args.output + f"mask_test{i}.jpg")
        
        x = np.sum(seg, axis=0)
        x = np.where((x!=0), 1, 0)
        x_min = np.argmax(x)
        x_max = len(x) - np.argmax(np.flip(x)) - 1
        
        y = np.sum(seg, axis=1)
        y = np.where((y!=0), 1, 0)
        y_min = np.argmax(y)
        y_max = len(y) - np.argmax(np.flip(y)) - 1
        
        new_results.append({'segmentation':seg, 'bounding_box':{'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}})
        print(new_results)
        
        draw.rectangle((x_min, y_max, x_max, y_min), outline=(0,255,0), width = 3)

input_image.save(args.output + f"test_1124.jpg")