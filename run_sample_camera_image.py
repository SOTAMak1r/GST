import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear,    'reset_parameters', lambda self: None) 
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None) 

from torchvision.utils import save_image
from torchvision import transforms

import numpy as np
import os
import cv2
import time
import argparse
from tokenizer.tokenizer_image.vq_model  import VQ_models
from tokenizer.tokenizer_camera.vq_model import CAM_models

from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from einops import rearrange, repeat
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.ray_utils import plucker_to_3d, create_ref_ray, plucker_to_camera, create_ref_camera
from utils.o3d_utils import save_images_and_cameras_from_cameras
from utils.image_utils import read_image_local


SPECIAL_TOKEN_IDX = {
    '<GEN>': 18432,
    '<EST>': 18433,
}

def empty_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.mkdir(folder_path)
        except:
            pass
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.image_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load camera model
    camera_model = CAM_models[args.cam_model](
        codebook_size=args.camera_codebook_size,
        codebook_embed_dim=args.camera_embed_dim,
    )
    camera_model.to(device)
    camera_model.eval()
    checkpoint = torch.load(args.camera_ckpt, map_location="cpu")
    camera_model.load_state_dict(checkpoint["model"])
    del checkpoint  
    print(f"camera tokenizer is loaded")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        qk_norm=args.qk_norm,
        special_token_num=args.special_token_num,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    model_weight = checkpoint["model"]
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    # load image
    cond_img = read_image_local(args.image_path, args.image_size, white_bg=True, transform=transform)
    cond_img = repeat(cond_img, "c h w -> b c h w", b=args.num_sample)
    cond_img = cond_img.to(device, non_blocking=True)
    _, _, [_, _, cond_indices] = vq_model.encode(cond_img)

    cond_x = cond_indices.reshape(cond_img.shape[0], -1)
    c_indices = cond_x # [B, 256]

    if args.special_token_num > 0:
        special_token = torch.ones((cond_x.shape[0], 1)).to(cond_x.dtype).to(cond_x.device) * SPECIAL_TOKEN_IDX['<GEN>']
        c_indices = torch.cat([c_indices, special_token], dim=-1)

    qzshape        = [int(cond_x.shape[0]), args.codebook_embed_dim, latent_size, latent_size]
    ref_qzshape    = [                   1, args.codebook_embed_dim, latent_size, latent_size]
    camera_qzshape = [int(cond_x.shape[0]), args.camera_embed_dim,   latent_size, latent_size]

    # begin sample
    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, (latent_size ** 2) * 2, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
    )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    

    # decoder 
    cameras = camera_model.decode_code(
        torch.clamp(index_sample[:, :(latent_size ** 2)]- args.codebook_size, 0, args.camera_codebook_size-1), 
        camera_qzshape
    )
    samples = vq_model.decode_code(    
        torch.clamp(index_sample[:, -(latent_size ** 2):], 0, args.codebook_size-1), 
        qzshape
    )

    empty_folder(args.sample_path)

    sample_paths = []
    for i in range(args.num_sample):
        sample_path = os.path.join(args.sample_path, f"sample_{i}.png")
        save_image(
            samples[i:i+1], sample_path, nrow=4, normalize=True, value_range=(-1, 1)
        )
        sample_paths.append(sample_path)
    
    # save GT
    samples = vq_model.decode_code(cond_x[:1], ref_qzshape) # output value is between [-1, 1]
    ref_sample_path = os.path.join(args.sample_path, f"sample_gt.png") 
    save_image(
        samples, ref_sample_path, nrow=4, normalize=True, value_range=(-1, 1)
    )
    sample_paths.append(ref_sample_path)

    # save ply
    B, _, H, W = cameras.shape
    c2ws = plucker_to_camera(cameras.permute(0, 2, 3, 1).reshape(B, -1, 6)) # [B, 3]    [B, 4096, 3] 
    c2w_ref = create_ref_camera()
    c2ws = np.concatenate([c2ws, c2w_ref], axis=0)

    save_images_and_cameras_from_cameras(
        c2ws,
        cameras.permute(0, 2, 3, 1).reshape(B, -1, 6).shape, 
        sample_paths,
        args.sample_path,
        camera_scale=args.ply_scale,
    )

    print('\nDONE !')




if __name__ == "__main__":
    from args import get_args
    args = get_args()

    with torch.no_grad():
        main(args)
