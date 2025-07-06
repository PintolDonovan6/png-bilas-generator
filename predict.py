from cog import BasePredictor, Input, Path
from PIL import Image
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.model = torch.load("weights.pth", map_location="cpu")
        self.model.eval()

    def predict(
        self,
        input_image: Path = Input(description="Input image for PNG pattern generation"),
        scale: float = Input(description="Optional scaling factor", default=1.5),
    ) -> Path:
        """Run a prediction and return the output image"""

        # Load input image
        img = Image.open(input_image).convert("RGB")

        # --------- Replace this block with your actual model logic ----------
        # Example dummy output: just saves the input image as output
        output_path = "output.png"
        img.save(output_path)
        # -------------------------------------------------------------------

        return Path(output_path)
