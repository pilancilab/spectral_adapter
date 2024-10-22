import json
import os

import torch  
from diffusers import DPMSolverMultistepScheduler

from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline
  
pretrained_model_path = "experiments/composed_edlora/chilloutmix/toys_r8/combined_model_base"
enable_edlora = True  # True for edlora, False for lora

pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
    new_concept_cfg = json.load(fr)
pipe.set_new_concept_cfg(new_concept_cfg)

TOK = 'a <happy1> <happy2> wearing a blue hat'
prompt = f'{TOK}'
negative_prompt = 'low quality, low resolution'

image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(0), guidance_scale=7.5).images[0]

image.save(f'experiments/happy_bluehat_seed0.jpg')