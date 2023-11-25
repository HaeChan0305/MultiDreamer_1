from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageChops, ImageOps
import numpy as np
import argparse
import os
import glob

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

    prompt = ""

    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    image = Image.open("/root/MultiDreamer/data/input/" + args.input).convert("RGB")

    for i, bbox_result in enumerate(eval(args.bbox)):
        print(">> bbox_result: ")
        print(bbox_result)
        mask_name1 = bbox_result['mask_name']
        input_bbox_image = Image.open("/root/MultiDreamer/data/output/" + output_folder_name + "/" + mask_name1 + ".jpg").convert("1")

        if i==0: mask_name2 = eval(args.bbox)[1]['mask_name']
        else: mask_name2 = eval(args.bbox)[0]['mask_name']

        bbox = bbox_result['bounding_box']

        bbox_img = Image.new('L', image.size)
        mask_array = np.array(bbox_img)
        mask_array[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = 255
        modified_bbox_img = Image.fromarray(mask_array).convert("1")

        seg_img = Image.open("/root/MultiDreamer/data/output/" + output_folder_name + "/" + mask_name2 + ".jpg").convert("1")

        intersection_image = ImageChops.logical_and(modified_bbox_img, seg_img).convert("L")

        input_bbox_image = input_bbox_image.point(lambda p: p > 128 and 255)
        result_image = Image.new("RGB", image.size, (0, 0, 0))
        result_image.paste(image, mask=input_bbox_image)
        result_image = result_image.convert("RGB")

        result_image.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/input{i}.jpg")

        image_result = pipe(prompt=prompt, image=result_image, mask_image=intersection_image).images[0]
        
        result_image.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/result_image{i}.jpg")
        image_result.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/inpainting{i}.jpg")
        intersection_image.save("/root/MultiDreamer/data/output/" + output_folder_name + f"/mask_intersection_test{i}.jpg")

if __name__ == "__main__":
    main()