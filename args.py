import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_camera.vq_model import CAM_models
from autoregressive.models.gpt import GPT_models

def get_args():
    parser = argparse.ArgumentParser()
    
    # auto-regressive model
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XXL")
    parser.add_argument("--gpt-ckpt",  type=str, default=None)
    parser.add_argument("--gpt-type",  type=str, default="unidata")  
    parser.add_argument("--vocab-size", type=int, default=18434, help="vocabulary size of visual tokenizer")
    parser.add_argument("--qk-norm", action='store_false')
    parser.add_argument("--special-token-num", type=int, default=1, help="number of special token")
    parser.add_argument("--cls-token-num", type=int, default=257, help="max token number of condition input")
    parser.add_argument("--max-seq-len",   type=int, default=512, help="max token number")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    # image tokenizer
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--image-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")

    # camera tokenizer
    parser.add_argument("--camera-ckpt", type=str, default=None, help="ckpt path for camera model")
    parser.add_argument("--cam-model", type=str, choices=list(CAM_models.keys()), default="VQ-4")
    parser.add_argument("--camera-codebook-size", type=int, default=2048, help="codebook size for camera model")
    parser.add_argument("--camera-embed-dim", type=int, default=4, help="codebook dimension for camera model")

    # train config
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--downsample-size-camera", type=int, default=4)

    # test config
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=100, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--num-sample", type=int, default=32)
    parser.add_argument("--image-path",  type=str, default='./assets/scene.jpg', help="observation path")
    parser.add_argument("--image-path-cond",   type=str, default='./assets/pair_cond.jpg',   help="observation path")
    parser.add_argument("--image-path-target", type=str, default='./assets/pair_target.jpg', help="observation path")
    parser.add_argument("--sample-path",  type=str, default='./sample', help="path to save samples")
    parser.add_argument("--ply-scale", type=float, default=0.3, help="to rescale the ply")


    args = parser.parse_args()
    return args