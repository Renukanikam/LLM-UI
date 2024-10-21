from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model for text-to-image generation
sd_model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
sd_model.to("cuda")

# Generate image from text
def generate_image_from_prompt(prompt):
    image = sd_model(prompt).images[0]
    return image
