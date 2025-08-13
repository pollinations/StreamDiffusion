import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
base_model_sdxl = "stabilityai/sdxl-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo & SDXL-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > or 
    <a
    href="https://huggingface.co/stabilityai/sdxl-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SDXL-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        model: str = Field(
            "sdxl-turbo",
            title="Model",
            field="select",
            id="model",
            info="Choose between SD-Turbo and SDXL-Turbo",
            values=["sdxl-turbo", "sd-turbo"],
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            768, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        self.current_model = "sdxl-turbo"
        self.args = args
        self.device = device
        self.torch_dtype = torch_dtype
        self.last_prompt = default_prompt
        
        # Initialize with SDXL-Turbo by default
        self._init_model("sdxl-turbo")

    def _init_model(self, model_type: str):
        """Initialize the StreamDiffusion model based on the selected type."""
        if model_type == "sdxl-turbo":
            model_path = base_model_sdxl
            t_index_list = [1]
            num_inference_steps = 2
            guidance_scale = 0.0
            cfg_type = "self"
        else:  # sd-turbo
            model_path = base_model
            t_index_list = [25, 30]
            num_inference_steps = 50
            guidance_scale = 1.2
            cfg_type = "none"
            
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=model_path,
            use_tiny_vae=self.args.taesd,
            device=self.device,
            dtype=self.torch_dtype,
            t_index_list=t_index_list,
            frame_buffer_size=1,
            width=512,
            height=768,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=self.args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type=cfg_type,
            use_safety_checker=self.args.safety_checker,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir=self.args.engine_dir,
        )
        
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        self.current_model = model_type

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Check if model needs to be switched
        if params.model != self.current_model:
            print(f"Switching from {self.current_model} to {params.model}...")
            self._init_model(params.model)
            
        # Check if prompt changed and needs re-preparation
        if params.prompt != self.last_prompt:
            if self.current_model == "sdxl-turbo":
                self.stream.prepare(
                    prompt=params.prompt,
                    negative_prompt=default_negative_prompt,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                )
            else:  # sd-turbo
                self.stream.prepare(
                    prompt=params.prompt,
                    negative_prompt=default_negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=1.2,
                )
            self.last_prompt = params.prompt
            
        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
