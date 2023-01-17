from typing import  Dict, List, Any
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type != 'cuda':
    raise ValueError("need to run on GPU")

class EndpointHandler():
    def __init__(self, path=""):
        # load the optimized model
        self.pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)


    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`dict`:. base64 encoded image
        """
        inputs = data.pop("inputs", data)
        
        # run inference pipeline
        with autocast(device.type):
            image = self.pipe(inputs, guidance_scale=7.5)["sample"][0]  
            
        # encode image as base 64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        # postprocess the prediction
        return {"image": img_str.decode()}