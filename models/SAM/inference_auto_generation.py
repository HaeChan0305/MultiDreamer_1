import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
import os

from tasks import interactive_infer_image_idino_m2m_auto

@torch.no_grad()
def inference(image, output_dir, level=1, *args, **kwargs):
    level = [level]
    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh='','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False
        model=model_sam
        results = interactive_infer_image_idino_m2m_auto(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, *args, **kwargs)

    new_results = []
    for result in results:
        seg = result['segmentation']
        if seg[0][0]:
            continue
        
        else:
            seg = seg.astype(np.uint8) * 255
            segmentation = Image.fromarray(seg)
            segmentation = segmentation.resize(input_image.size)
            seg = np.array(segmentation)
            
            x = np.sum(seg, axis=0)
            x = np.where((x!=0), 1, 0)
            x_min = np.argmax(x)
            x_max = len(x) - np.argmax(np.flip(x)) - 1
            
            y = np.sum(seg, axis=1)
            y = np.where((y!=0), 1, 0)
            y_min = np.argmax(y)
            y_max = len(y) - np.argmax(np.flip(y)) - 1
            
            new_results.append({'segmentation':segmentation, 'bounding_box':{'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}})
    
    return new_results

def calculate_bbox_area(bbox):
    return (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser('SemanticSAM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/semantic_sam_only_sa-1b_swinL.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--ckpt', default="models/swinl_only_sam_many2many.pth", metavar="FILE", help='path to ckpt', )
    parser.add_argument('--level', default=2, )
    parser.add_argument('--input', default="ref_washing.png", help='input image should be in data/input folder', )
    parser.add_argument('--output_dir', default="ref_washing.png", help='output image should be in data/output folder', )
    args = parser.parse_args()
    
    # Load Model
    cur_model = 'None'
    model=None
    model_size=None
    ckpt=None
    cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
        'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

    sam_cfg=cfgs['L']
    opt = load_opt_from_config_file(sam_cfg)
    model_sam = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()

    # Set pathes
    input_image = Image.open(args.input).convert("RGB")
    os.makedirs(args.output_dir, exist_ok=True)

    # Inference
    draw = ImageDraw.Draw(input_image)
    results = inference(input_image, args.output_dir, args.level, draw)
    
    # Filtering
    if len(results) < 2:
        raise Exception(f"Detected Only {len(results)} Bounding Box")
    else:
        results = sorted(results, key=lambda x: calculate_bbox_area(x['bounding_box']))
        results = results[:2]
    
    # Save
    for i, result in enumerate(results):
        result["segmentation"].save(os.path.join(args.output_dir, f"mask{i}.jpg"))
        b = result['bounding_box']
        draw.rectangle((b["x_min"], b["y_max"], b["x_max"], b["y_min"],), outline=(0,255,0), width = 3)
    input_image.save(os.path.join(args.output_dir, f"bbox.jpg"))
        
    print([result['bounding_box'] for result in results])
    