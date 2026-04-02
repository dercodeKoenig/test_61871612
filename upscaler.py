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


model_id = 'microsoft/Florence-2-large'
analyzer_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Florence 2 loaded successfully!")

supir_model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign="Q")
supir_model.half()
supir_model.ae_dtype = convert_dtype("bf16")
supir_model.model.dtype = convert_dtype("fp16")
print("SUPIR model loaded successfully!")

supir_model_tiled = create_SUPIR_model('options/SUPIR_v0_tiled.yaml', SUPIR_sign="Q")
supir_model_tiled.half()
supir_model_tiled.ae_dtype = convert_dtype("bf16")
supir_model_tiled.model.dtype = convert_dtype("fp16")
supir_model_tiled.init_tile_vae(encoder_tile_size=3072, decoder_tile_size=192)
print("Tiled SUPIR model loaded successfully!")


def get_supir_model(tiled):
    if(tiled):
        supir_model.cpu()
        torch.cuda.empty_cache()
        return supir_model_tiled.cuda()
    else:
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
        task=prompt, # Changed from task_prompt to prompt
        image_size=(image.width, image.height)
    )

    analyzer_model.cpu()
    torch.cuda.empty_cache()
    return parsed_answer

def upscale(img, scale, captions, restoration = False):

    LQ_ips = img
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, scale, min_size=256)
    LQ_img = LQ_img.unsqueeze(0).to("cuda")[:, :3, :, :]

    print(img.size, scale, LQ_img.shape, captions)

    t1 = time.time()


    _, _, h_scaled, w_scaled = LQ_img.shape
    needs_tiling = False
    if (w_scaled * h_scaled) > (2000 * 3000) and w_scaled >= 1024 and h_scaled >= 1024:
        print(f"--- VRAM Protection: Using Tiled Model ({w_scaled}x{h_scaled}) ---")
        needs_tiling = True

    model = get_supir_model(tiled=needs_tiling)    

    control_scale = 0.9
    cfg_scale = 4.0     
    s_noise = 1.02
    cfg_scale_start = 2

    restoration_scale = -1
    if restoration:
        restoration_scale = 5

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
            p_p='clean, high quality, highly detailed, sharp focus',
            n_p='blurry, lowres, grainy, distorted, wavy lines, artifacts',
            color_fix_type='Wavelet',
            use_linear_CFG=True,
            cfg_scale_start=cfg_scale_start,
            control_scale_start=0
        )


    t2 = time.time()
    print("upscale done in ", t2 - t1)

    return Tensor2PIL(samples[0], h0, w0)


def enhance(image, scale):

    description = run_analysis("<DETAILED_CAPTION>", image)["<DETAILED_CAPTION>"]
    caption = run_analysis("<CAPTION>", image)["<CAPTION>"]
    print(description)
    print(caption)
    print(image.size)
    print(scale)

    img_size = image.width * image.height

    results = run_analysis('<DENSE_REGION_CAPTION>', image)
    
    cropped_images = []
    bboxes = results['<DENSE_REGION_CAPTION>']['bboxes']
    labels = results['<DENSE_REGION_CAPTION>']['labels']
    
    pad_val = 0.4
    
    for i, bbox in enumerate(bboxes):
        x1_orig, y1_orig, x2_orig, y2_orig = [int(coord) for coord in bbox]
    
        # Calculate width and height of the original bounding box
        width = x2_orig - x1_orig
        height = y2_orig - y1_orig
        if(width < 60 or height < 60):
            # too small
            continue 
        if(width * height > img_size * 0.9):
            # basically its the entire image - skip
            continue
    
        # Calculate padding amount
        padding_x = int(pad_val * width)
        padding_y = int(pad_val * height)
    
        # Apply padding and ensure coordinates stay within image boundaries
        x1_padded = max(0, x1_orig - padding_x)
        y1_padded = max(0, y1_orig - padding_y)
        x2_padded = min(image.width, x2_orig + padding_x)
        y2_padded = min(image.height, y2_orig + padding_y)
    
        cropped_image = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
        cropped_images.append({'image': cropped_image, 'label': labels[i], 'original_bbox': [x1_orig, y1_orig, x2_orig, y2_orig]})
    
    print(f"Found {len(cropped_images)} dense regions.")
    
    # upscale crops
    upscaled_crops = []
    for i, item in enumerate(cropped_images):
        print(f"Cropped Image {i+1}: {item['label']} (Original BBox: {item['original_bbox']})")
        #desc2 = run_analysis('<DETAILED_CAPTION>', item['image'])['<DETAILED_CAPTION>']
        out = upscale(item['image'], scale, description)
        new_entry = {"image" : out, "original_bbox" : item["original_bbox"], "label" : item['label']}
        upscaled_crops.append(new_entry)
    
    
    # upscale main / background
    print("Upscaling main image now...")
    upscaled = upscale(image, scale, description)
    
    
    
    def create_smart_feathered_mask(crop_size, actual_pads, scale):
        """
        Creates a mask that only feathers sides where padding actually exists.
        actual_pads = (left, top, right, bottom) in original pixels.
        """
        mask = Image.new("L", crop_size, 0)
        draw = ImageDraw.Draw(mask)
    
        # Scale the actual padding used
        pl = int(actual_pads[0] * scale)
        pt = int(actual_pads[1] * scale)
        pr = int(actual_pads[2] * scale)
        pb = int(actual_pads[3] * scale)
    
        w, h = crop_size
    
        # The opaque area is the crop minus the scaled padding
        # We use a small 2-pixel buffer to ensure the 'core' is solid
        inner_box = [pl, pt, w - pr, h - pb]
        draw.rectangle(inner_box, fill=255)
    
        # Calculate blur radius based on the average padding
        # If padding is 0 (at edge), it doesn't contribute to the blur
        valid_pads = [p for p in [pl, pt, pr, pb] if p > 0]
        blur_radius = max(1, sum(valid_pads) // len(valid_pads) // 2) if valid_pads else 1
    
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    
        # CRITICAL: If a side had 0 padding (it's the image edge), 
        # we must force it back to 255 so the edge isn't blurry/transparent.
        draw_solid = ImageDraw.Draw(mask)
        if pl == 0: draw_solid.rectangle([0, 0, blur_radius, h], fill=255)
        if pt == 0: draw_solid.rectangle([0, 0, w, blur_radius], fill=255)
        if pr == 0: draw_solid.rectangle([w - blur_radius, 0, w, h], fill=255)
        if pb == 0: draw_solid.rectangle([0, h - blur_radius, w, h], fill=255)
    
        return mask
    
    # --- 1. The Painter's Algorithm (Sorting) ---
    # Sort by area descending: large background chunks first, tiny high-detail faces last.
    upscaled_crops.sort(key=lambda x: (x['original_bbox'][2] - x['original_bbox'][0]) * (x['original_bbox'][3] - x['original_bbox'][1]), 
                        reverse=True)
    
    final_image = upscaled.copy()
    
    for item in upscaled_crops:
        print("stitching", item["label"])
        crop_img = item['image']
        x1_orig, y1_orig, x2_orig, y2_orig = item['original_bbox']
    
        # Calculate dimensions
        bw, bh = x2_orig - x1_orig, y2_orig - y1_orig
        px, py = int(pad_val * bw), int(pad_val * bh)
    
        # Re-calculate the actual padded coordinates used in the crop phase
        x1_padded = max(0, x1_orig - px)
        y1_padded = max(0, y1_orig - py)
        x2_padded = min(image.width, x2_orig + px)
        y2_padded = min(image.height, y2_orig + py)
    
        # Calculate the actual amount of padding that was possible
        actual_pads = (
            x1_orig - x1_padded, # left
            y1_orig - y1_padded, # top
            x2_padded - x2_orig, # right
            y2_padded - y2_orig  # bottom
        )
    
        # Map to upscaled space
        target_pos = (int(x1_padded * scale), int(y1_padded * scale))
    
        # Generate the edge-aware mask
        mask = create_smart_feathered_mask(crop_img.size, actual_pads, scale)
    
        # Paste
        final_image.paste(crop_img, target_pos, mask)
    
    print("Stitching complete!")

    refined = upscale(final_image, 1, description, True)

    print("refinement complete")
    print("")
    print("")
    
    return upscaled, refined
    