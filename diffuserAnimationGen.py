import torch
from diffusers import AutoencoderKL, ControlNetModel, MotionAdapter, AnimateDiffPipeline, EulerDiscreteScheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image

import numpy as np
import cv2

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from compel import Compel, DiffusersTextualInversionManager


def initializeDiffusersAnimate():
    motion_id = "guoyww/animatediff-motion-adapter-v1-5-2"
    adapter = MotionAdapter.from_pretrained(motion_id)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

    # model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

    model_id = "emilianJR/epiCRealism" 
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        controlnet=controlnet,
        vae=vae,
        custom_pipeline="pipeline_animatediff_controlnet",
    ).to(device="cuda", dtype=torch.float16)

    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler", beta_schedule="linear", clip_sample=False, timestep_spacing="linspace", steps_offset=1, final_sigmas_type = "sigma_min"
    )

    pipe.enable_vae_slicing()

    pipe.load_textual_inversion("sd-concepts-library/ali-1-4-CCTV")

    return pipe


def initializeDiffusersAnimate_v2():
    device = "cuda"
    dtype = torch.float16

    step = 4  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    pipe.enable_vae_slicing()

    return pipe



def generateAnimate(diffuser_pipe,conditioning_frames, prompt, negative_prompt="", inf_steps = 20):
    

    textual_inversion_manager = DiffusersTextualInversionManager(diffuser_pipe)
    compel_proc = Compel(
        tokenizer=diffuser_pipe.tokenizer,
        text_encoder=diffuser_pipe.text_encoder,
        textual_inversion_manager=textual_inversion_manager)

    prompt_embeds = compel_proc(prompt)

    num_frames_gen = len(conditioning_frames)

    result = diffuser_pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt=negative_prompt,
        width=320,
        height=800,
        conditioning_frames=conditioning_frames,
        num_inference_steps=inf_steps,
        num_frames = num_frames_gen
    ).frames[0]


    return result