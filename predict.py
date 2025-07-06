import torch
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(self, prompt: str = Input(description="Input prompt")):
        image = self.pipe(prompt).images[0]
        image.save("output.png")
        return "output.png"
