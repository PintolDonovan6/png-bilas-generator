import argparse
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def main(prompt):
    model_id = "runwayml/stable-diffusion-v1-5"  # popular stable diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)

    # Generate image
    image = pipe(prompt, guidance_scale=7.5).images[0]

    # Save image locally
    output_path = "output.png"
    image.save(output_path)

    print(output_path)  # Cog uses this to get the output path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    main(args.prompt)
