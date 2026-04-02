import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
import time
import numpy as np
import cv2
import torch
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFilter
import gc
allow_tiled = False

model_id = 'microsoft/Florence-2-large'
analyzer_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Florence 2 loaded successfully!")

supir_model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign="Q")
supir_model.half()
supir_model.ae_dtype = convert_dtype("bf16")
supir_model.model.dtype = convert_dtype("fp16")
print("SUPIR model loaded successfully!")

supir_model_tiled = None
if allow_tiled:
    supir_model_tiled = create_SUPIR_model('options/SUPIR_v0_tiled.yaml', SUPIR_sign="Q")
    supir_model_tiled.half()
    supir_model_tiled.ae_dtype = convert_dtype("bf16")
    supir_model_tiled.model.dtype = convert_dtype("fp16")
    supir_model_tiled.init_tile_vae(encoder_tile_size=3072, decoder_tile_size=192)
    print("Tiled SUPIR model loaded successfully!")


def get_supir_model(tiled):
    if tiled:
        supir_model.cpu()
        torch.cuda.empty_cache()
        return supir_model_tiled.cuda()
    else:
        if supir_model_tiled is not None:
            supir_model_tiled.cpu()
        torch.cuda.empty_cache()
        return supir_model.cuda()

def run_analysis(prompt, image):
    analyzer_model.cuda()
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = analyzer_model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=200,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    analyzer_model.cpu()
    torch.cuda.empty_cache()
    return parsed_answer

def upscale(img, scale, captions):

    LQ_ips = img
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, scale, min_size=512)
    LQ_img = LQ_img.unsqueeze(0).to("cuda")[:, :3, :, :]

    if captions:
        # Remove any trailing dots, commas, or spaces, then add a clean comma-space
        # the program will just append prompt and p_p together
        captions = captions.rstrip('. ,') + ', '
    else:
        captions = "" # Handle the empty case safely
    
    print(img.size, scale, LQ_img.shape, captions)

    t1 = time.time()

    _, _, h_scaled, w_scaled = LQ_img.shape
    needs_tiling = False
    if allow_tiled:
        if (w_scaled * h_scaled) > (2048 * 3072) and w_scaled >= 1024 and h_scaled >= 1024:
            print(f"--- VRAM Protection: Using Tiled Model ({w_scaled}x{h_scaled}) ---")
            needs_tiling = True

    model = get_supir_model(tiled=needs_tiling)    

    control_scale = 0.98
    cfg_scale = 3.0   
    s_noise = 0.98
    cfg_scale_start = 1.2
    color_fix_type='Wavelet'
    restoration_scale = -1

    with torch.no_grad():
        samples = model.batchify_sample(
            LQ_img,
            [captions],
            num_steps=32,
            restoration_scale=restoration_scale,
            s_churn=0,
            s_noise=s_noise,
            cfg_scale=cfg_scale,
            control_scale=control_scale,
            seed=random.randint(0, 999999),
            num_samples=1,
            p_p='high quality, detailed, cinematic, perfect without deformations',
            n_p='blurry, messy, noisy, deformed, lowres, low quality, grainy, distorted, artifacts, bad anatomy, malformed hands, malformed fingers, malformed teeth',
            color_fix_type=color_fix_type,
            use_linear_CFG=True,
            cfg_scale_start=cfg_scale_start,
            control_scale_start=0
        )


    t2 = time.time()
    print("upscale done in ", t2 - t1)

    return Tensor2PIL(samples[0], h0, w0)


def enhance(image, scale, description = None):

    if description is None:
        #description = run_analysis("<DETAILED_CAPTION>", image)["<DETAILED_CAPTION>"]
        description = run_analysis("<CAPTION>", image)["<CAPTION>"]
        print("generated description:", description)

    upscaled = upscale(image, scale, description)
    return upscaled
    #return image
    