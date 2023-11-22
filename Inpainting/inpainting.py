from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    # revision="fp16",
    # torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
# pipe.enable_model_cpu_offload()

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is

prompt = "background"
image = Image.open("/root/MultiDreamer/ROCA2/network/assets/sofa.jpg")
mask_image = Image.open("/root/MultiDreamer/ROCA2/network/mask_image2.jpg")

image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./inpainted.png")