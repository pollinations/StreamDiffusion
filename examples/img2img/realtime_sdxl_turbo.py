import os
import sys
from typing import Literal, Dict, Optional

import fire
import cv2
from PIL import Image
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

DEFAULT_PROMPT = "Portrait of The Joker in a Halloween costume, face painting, with a glare pose, detailed, intricate, full of color, cinematic lighting, trending on ArtStation, 8K, hyperrealistic, focused, extreme details, Unreal Engine 5 cinematic, masterpiece"
NEGATIVE_PROMPT = "black and white, blurry, low resolution, pixelated, pixel art, low quality, low fidelity"

def main(
    model_id_or_path: str = "stabilityai/sdxl-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = DEFAULT_PROMPT,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    use_denoising_batch: bool = True, 
    seed: int = 2,
):
    
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------    
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optionalq
        The seed, by default 2. if -1, use random seed.
    """

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[1],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        use_lcm_lora=False,
        vae_id=None,
        acceleration=acceleration,
        output_type="np",
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="self",   
        seed=seed,
        #engine_dir="../../engines",
    )
    
    stream.prepare(
        prompt=prompt,
        negative_prompt= NEGATIVE_PROMPT,
        num_inference_steps=2,
        guidance_scale=0.0,
    )

    # Open a connection to the camera (you may need to change the camera index)
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate center crop parameters  
    crop_size = min(frame_width, frame_height)
    mid_x, mid_y = int(frame_width/2), int(frame_height/2)
    cw2, ch2 = int(crop_size/2), int(crop_size/2) 

    # Set up a window for displaying the generated images
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Generated Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Generated Image', 768, 768) 
    ret, frame = cap.read()
   
    # Initialize variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_display = "FPS: 0"
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Update FPS value once every second or after every 20 frames
        if elapsed_time >= 1.0 or frame_count == 20:
            fps = frame_count / elapsed_time
            fps_display = f"FPS: {fps:.2f}"
            frame_count = 0
            start_time = time.time()

        cropped_img = frame[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2] #center crop
        # Generate the image 
        input_image = Image.fromarray(cropped_img)
        input_image = stream.preprocess_image(input_image)
        output_image = stream(image=input_image)

        # Display the generated image
        cv2.imshow('Generated Image', output_image)
        # Display the orijinal image
        cv2.putText(cropped_img, fps_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Original Image', cropped_img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fire.Fire(main)
