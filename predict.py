import torch
from diffusers import StableDiffusionPipeline

class Predictor:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            revision="fp16"
        ).to("cuda")

    def predict(self, prompt: str) -> str:
        image = self.pipe(prompt).images[0]
        image.save("/tmp/output.png")
        return "/tmp/output.png"
