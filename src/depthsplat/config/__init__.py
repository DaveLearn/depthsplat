
from typing import Literal

import torch
from ..model.encoder import EncoderDepthSplat, EncoderDepthSplatCfg
from dacite import Config, from_dict
from omegaconf import OmegaConf
from pathlib import Path
from huggingface_hub import hf_hub_download


def load_encoder_config(cfg_path: str) -> EncoderDepthSplatCfg:
    return from_dict(
    EncoderDepthSplatCfg,
    OmegaConf.to_container(OmegaConf.load(cfg_path)),
    config=Config(type_hooks={**{Path: Path}}),
)

def init_pretrained_encoder_depth_only(size: Literal["small", "base", "large"]) -> EncoderDepthSplat:
    
    cfg = load_encoder_config(Path(__file__).parent / "encoder/depthsplat.yaml")

    cfg.return_depth = True
    cfg.train_depth_only = True

    if size == "large":
        cfg.num_scales = 2
        cfg.upsample_factor = 4
        cfg.lowest_feature_resolution = 8
        cfg.monodepth_vit_type="vitl"
    
        pretrained_path = download_pretrained("depthsplat-depth-large-50d3d7cf.pth")
        
    if size == "small":
        cfg.upsample_factor = 8
        cfg.lowest_feature_resolution = 8
        
        pretrained_path = download_pretrained("depthsplat-depth-small-3d79dd5e.pth")

    if size == "base":
        cfg.num_scales = 2
        cfg.upsample_factor = 4
        cfg.lowest_feature_resolution = 8
        cfg.monodepth_vit_type="vitb"

        pretrained_path = download_pretrained("depthsplat-depth-base-f57113bd.pth")


    state_dict = torch.load(pretrained_path, weights_only=True)
    encoder = EncoderDepthSplat(cfg)
    encoder.depth_predictor.load_state_dict(state_dict["model"])

    return encoder

def init_pretrained_encoder(size: Literal["large", "base"]) -> EncoderDepthSplat:
    
    cfg = load_encoder_config(Path(__file__).parent / "encoder/depthsplat.yaml")

    if size == "large":
        path = "depthsplat-gs-large-re10k-256x256-288d9b26.pth"
        cfg.num_scales=2 
        cfg.upsample_factor=2 
        cfg.lowest_feature_resolution=4 
        cfg.monodepth_vit_type="vitl" 
        cfg.gaussian_regressor_channels=64 
        cfg.color_large_unet= True 
        cfg.feature_upsampler_channels=128 
    elif size == "base":
        path = "depthsplat-gs-base-re10k-256x256-044fdb17.pth"
        cfg.num_scales=2 
        cfg.upsample_factor=2 
        cfg.lowest_feature_resolution=4 
        cfg.monodepth_vit_type="vitb" 
        cfg.gaussian_regressor_channels=32 
        cfg.color_large_unet= True 
        cfg.feature_upsampler_channels=128 
    
    path = download_pretrained(path)
    
    encoder = EncoderDepthSplat(cfg)
    state_dict = torch.load(path, weights_only=True)
    # state keys are all prefixed with "encoder" so we need to remove that prefix
    state_dict2 = {k.replace("encoder.", ""): v for k, v in state_dict['state_dict'].items()}
    encoder.load_state_dict(state_dict2)
    return encoder

def download_pretrained(filename: str):
    return hf_hub_download(repo_id="haofeixu/depthsplat", filename=filename, local_dir="pretrained")