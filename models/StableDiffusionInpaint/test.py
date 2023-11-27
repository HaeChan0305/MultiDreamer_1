from PIL import Image, ImageDraw
import numpy as np

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

mask_img = Image.open("/root/MultiDreamer/data/output/33_1/mask_intersection1.jpg")
mask_img.save("/root/MultiDreamer/models/StableDiffusionInpaint/outputs/mask_input.jpg")
expanded_mask_img = expand_mask(mask_img)
expanded_mask_img.save("/root/MultiDreamer/models/StableDiffusionInpaint/outputs/expanded_mask.jpg")