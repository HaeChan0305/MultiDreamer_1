from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageChops, ImageDraw, ImageOps
import numpy as np
import argparse
import os
import glob
import re

def calculate_bbox_area(bbox):
    return (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])

def expand_mask(mask_image):
    mask = np.array(mask_image)
    x = np.sum(mask, axis=0)
    x = np.where((x!=0), 1, 0)
    x_min = np.argmax(x)
    x_max = len(x) - np.argmax(np.flip(x)) - 1
    
    y = np.sum(mask, axis=1)
    y = np.where((y!=0), 1, 0)
    y_min = np.argmax(y)
    y_max = len(y) - np.argmax(np.flip(y)) - 1
    
    expanded_mask_image = Image.new('L', mask_image.size, color=0)
    draw = ImageDraw.Draw(expanded_mask_image)
    draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
    
    return expanded_mask_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inpainting Demo', add_help=False)
    parser.add_argument('--input', default="ref_washing.png", help='input image should be in data/input folder', )
    parser.add_argument('--output_dir', default="ref_washing.png", help='output_dir image should be in data/input folder', )
    parser.add_argument('--bbox')
    args = parser.parse_args()

    # bounding_box_results = os.environ.get('BBOX')
    print(">> BBOX result: " + args.bbox)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        # revision="fp16",
        # torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    prompt = ""

    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    image = Image.open(args.input).convert("RGB")
    
    bbox_list = eval(args.bbox)

    for i, bbox in enumerate(bbox_list):
        print(">> bbox_result: ")
        print(bbox)
        input_mask_image = Image.open(os.path.join(args.output_dir,  f"mask{i}.jpg")).convert("1")

        # inpainting의 mask로 사용할 intersection region 구하기
        bbox_img = Image.new('L', image.size)
        mask_array = np.array(bbox_img)
        mask_array[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = 255
        modified_bbox_img = Image.fromarray(mask_array).convert("1")

        seg_img = Image.open(os.path.join(args.output_dir, f"mask{1-i}.jpg")).convert("1")

        intersection_image = ImageChops.logical_and(modified_bbox_img, seg_img).convert("L")
        intersection_image = expand_mask(intersection_image)
        
        # input image에서 각각의 mask 모양대로 자르기
        input_mask_image = input_mask_image.point(lambda p: p > 128 and 255)
        result_image = Image.new("RGB", image.size, (0, 0, 0))
        result_image.paste(image, mask=input_mask_image)
        result_image = result_image.convert("RGB")

        # Inpainting
        inpainting_result = pipe(prompt=prompt, image=result_image, mask_image=intersection_image).images[0]
        inpainting_result = inpainting_result.convert("RGBA")
        
        # Save
        result_image.save(os.path.join(args.output_dir, f"input{i}.jpg"))
        inpainting_result.save(os.path.join(args.output_dir, f"inpainting{i}.png"))
        intersection_image.save(os.path.join(args.output_dir, f"mask_intersection{i}.jpg"))
