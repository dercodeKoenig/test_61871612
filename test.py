import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import queue
import random
import threading
import time
from io import BytesIO
import numpy as np
import cv2
import requests
from queue import Queue
import torch
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
import gc


model = create_SUPIR_model('options/SUPIR_v0_tiled.yaml', SUPIR_sign="Q")
model.half()
model.ae_dtype = convert_dtype("bf16")
model.model.dtype = convert_dtype("fp16")
model.init_tile_vae(encoder_tile_size=3072, decoder_tile_size=192)
model.to("cuda")
print("SUPIR model loaded successfully!")




img = Image.open("baboon.png")
img = img.convert("RGB")

img_size = img.width * img.height
target_size = 4096 * 4096
scale = math.sqrt(target_size / img_size)


LQ_ips = img
LQ_img, h0, w0 = PIL2Tensor(LQ_ips, scale, min_size=512)
LQ_img = LQ_img.unsqueeze(0).to("cuda")[:, :3, :, :]

captions = [""]

# Diffusion Process
print(img.size, scale, LQ_img.shape, captions)

extra_creativity = 0.1
control_scale = 1.0 - 0.2 * extra_creativity
cfg_scale = 2.0 + 6.0 * extra_creativity
s_noise = 0.98 + 0.04 * extra_creativity
cfg_scale_start = 1.0 + 2.0 * extra_creativity

t1 = time.time()

with torch.no_grad():
    samples = model.batchify_sample(
        LQ_img,
        captions,
        num_steps=6,
        restoration_scale=-1,
        s_churn=0,
        s_noise=s_noise,
        cfg_scale=cfg_scale,
        control_scale=control_scale,
        seed=random.randint(0, 999999),
        num_samples=1,
        p_p='clean, high quality, detailed',
        n_p='bad quality, blurry, deformed, noise, grainy, distorted, malformed hands, extra limbs, missing limbs, bad anatomy',
        color_fix_type='Wavelet',
        use_linear_CFG=True,
        use_linear_control_scale=False,
        cfg_scale_start=cfg_scale_start,
        control_scale_start=0
    )


t2 = time.time()
print("upscale done:", t2-t1)

# Convert tensor back to PIL
upscaled = Tensor2PIL(samples[0], h0, w0)

upscaled.save("upscaled.png")