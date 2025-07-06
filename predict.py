from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            revision="fp16"
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="PNG design prompt, e.g., 'bilas pattern with traditional Kundu drum colors'")
    ) -> Path:
        image = self.pipe(prompt).images[0]
        output_path = "/tmp/png-bilas.png"
        image.save(output_path)
        return Path(output_path)
