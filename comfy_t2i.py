import sys
import time

sys.path.append("ComfyUI")
import numpy as np
import torch
from comfy.sd import load_checkpoint_guess_config
from comfy_extras.nodes_canny import Canny
from image_utils import load_image
# load controlnet controlnet loader
from nodes import (CheckpointLoaderSimple, CLIPTextEncode,
                   ControlNetApplyAdvanced, ControlNetLoader, EmptyLatentImage,
                   KSampler, LoadImage, VAEDecode)
from PIL import Image

# print(KSampler.INPUT_TYPES())
# for i in range(7):
#     t = time.time()
    
#     x = load_controlnet("/home/erwann/ComfyUI/models/controlnet/canny.safetensors", None)
#     print(time.time() - t)
# print(x)


#cnet

# canny preprocessir

# torch.backends.cuda.enable_math_sdp(True)
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)

# model = CheckpointLoaderSimple().load_checkpoint("/home/erwann/Fooocus/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors")
model, clip, vae, _ = load_checkpoint_guess_config("/home/erwann/Fooocus/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors")

text_encoder = CLIPTextEncode()
vae_decoder = VAEDecode()
sampler = KSampler()


from contexttimer import Timer
# with torch.cuda.amp.autocast():
from line_profiler import LineProfiler

# lp = LineProfiler()

# @lp

def load_lora(model, clip, lora_path, strength):
    if lora_path is None:
        return (model, clip)
    import comfy
    print('loading lora')
    print(f"lora_path = {lora_path}")
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
    return (model_lora, clip_lora)


def t2i(prompt, lora_path=None):
    global model, clip, vae, cnet, canny_processor, text_encoder, vae_decoder, sampler
    pos = prompt
    neg = ""
    model, clip = load_lora(model, clip, lora_path, 1)
    with torch.inference_mode(), Timer() as t:
        pos = text_encoder.encode(clip, pos)[0]
        neg = text_encoder.encode(clip, neg)[0]

        empty_latent = EmptyLatentImage().generate(1024, 1024,)[0]

        sampler_args = {
            'model': model,
            'seed': 123,
            'steps': 35,
            'cfg': 8.,
            'sampler_name': 'euler',
            'scheduler': 'normal',
            'positive': pos,
            'negative': neg,
            'denoise': 1.,
            'latent_image': empty_latent,
        }
        latent = sampler.sample(**sampler_args)[0]
        images = vae_decoder.decode(vae, latent)[0]
        print("Time taken:", t.elapsed)
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save("test_out.png")
            return img
        print(t.elapsed)
    # lp.print_stats(output_unit=1.0)
# pos = 

if __name__ == "__main__":
    # t2i("jrch, puppy", lora_path="/home/erwann/ComfyUI/models/loras/lorajillus5/at-step00000800.safetensors")
    # t2i("jrch, puppy", lora_path="/home/erwann/ComfyUI/models/loras/lorajillus5/at-step00001200.safetensors")
    t2i("jrch, puppy", lora_path="/home/erwann/ComfyUI/models/loras/lorajillus9/at-step00001000.safetensors")
    # for i in range(4):
    #     infer()

# latent = sampler.sample(model=model, positive= )

# load checkpoint (ckptloadersimple)


# cnetapplyadncnead -> ksampler


