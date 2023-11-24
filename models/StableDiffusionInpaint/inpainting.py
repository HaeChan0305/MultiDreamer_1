from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageChops, ImageOps
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    # revision="fp16",
    # torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = ""
# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open("images/sofa_table.jpg").convert("RGB")
# mask_image = Image.open("outputs/sofa_table_mask1.jpg")

# mask_image = ImageOps.invert(mask_image)
# mask_array = np.array(mask_image)

# modified_mask_image = Image.fromarray(mask_array)
# modified_mask_image = modified_mask_image.resize(image.size)

mask2 = Image.new('L', image.size)
mask2_array = np.array(mask2)
mask2_array[500:1374, 300:1593] = 255
# mask2_array = ~mask2_array.astype(np.uint8)
modified_mask2_image = Image.fromarray(mask2_array)
# modified_mask2_image = modified_mask2_image.resize(image.size)

# modified_mask_image = modified_mask_image.convert("1")
# modified_mask2_image = modified_mask2_image.convert("1")
# image_mode = modified_mask_image.mode
# print(image_mode)
# image2_mode = modified_mask2_image.mode
# print(image2_mode)

# intersection_image = ImageChops.logical_and(modified_mask_image, modified_mask2_image)

# print(image.size)
# print(mask_image.size)

image = pipe(prompt=prompt, image=image, mask_image=modified_mask2_image).images[0]
modified_mask2_image.save("outputs/mask.jpg")
image.save("outputs/sofa_table_mask_1124.jpg")
