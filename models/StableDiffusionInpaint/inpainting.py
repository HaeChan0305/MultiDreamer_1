from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageChops, ImageOps
import numpy as np
import argparse
import os
import glob

def calculate_bbox_area(bbox):
    return (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])

def main():
    parser = argparse.ArgumentParser('Inpainting Demo', add_help=False)
    parser.add_argument('--input', default="ref_washing.png", help='input image should be in data/input folder', )
    parser.add_argument('--bbox')
    args = parser.parse_args()

    output_folder_name = args.input.split('.')[0]
    # bounding_box_results = os.environ.get('BBOX')
    print(">> BBOX result: " + args.bbox)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        # revision="fp16",
        # torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    prompt = "just the input image"

    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    image = Image.open("/root/MultiDreamer/data/input/eval/" + args.input).convert("RGB")
    
    bbox_list = eval(args.bbox)

    if len(bbox_list) == 1:
        print("There should be only 2 objects.")
        return
    elif len(bbox_list) > 2:
        # bounding box 넓이 기준으로 딕셔너리 정렬 후 작은 두 개만 남기기
        sorted_bbox_list = sorted(bbox_list, key=lambda x: calculate_bbox_area(x['bounding_box']))
        bbox_list = sorted_bbox_list[:2]

    for i, bbox_result in enumerate(bbox_list):
        print(">> bbox_result: ")
        print(bbox_result)
        mask_name1 = bbox_result['mask_name']
        input_mask_image = Image.open("/root/MultiDreamer/data/output/" + output_folder_name + "/" + mask_name1 + ".jpg").convert("1")

        if i==0: mask_name2 = bbox_list[1]['mask_name']
        else: mask_name2 = bbox_list[0]['mask_name']

        # inpainting의 mask로 사용할 intersection region 구하기
        bbox = bbox_result['bounding_box']

        bbox_img = Image.new('L', image.size)
        mask_array = np.array(bbox_img)
        mask_array[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = 255
        modified_bbox_img = Image.fromarray(mask_array).convert("1")

        seg_img = Image.open("/root/MultiDreamer/data/output/" + output_folder_name + "/" + mask_name2 + ".jpg").convert("1")

        intersection_image = ImageChops.logical_and(modified_bbox_img, seg_img).convert("L")

        # input image에서 각각의 mask 모양대로 자르기
        input_mask_image = input_mask_image.point(lambda p: p > 128 and 255)
        result_image = Image.new("RGB", image.size, (0, 0, 0))
        result_image.paste(image, mask=input_mask_image)
        result_image = result_image.convert("RGB")

        # inpainting
        inpainting_result = pipe(prompt=prompt, image=result_image, mask_image=intersection_image).images[0]
        inpainting_result = inpainting_result.convert("RGBA")
        
        result_image.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/input{i}.jpg")
        inpainting_result.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/inpainting{i}.png")
        intersection_image.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/mask_intersection{i}.jpg")

if __name__ == "__main__":
    main()