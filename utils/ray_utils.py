import torch
import math
from kiui.op import safe_normalize
import torch.nn.functional as F
import numpy as np


def get_rays(pose, h, w, fovy=49.1, opengl=False):
    x, y = torch.meshgrid(
        torch.arange(w),
        torch.arange(h),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    pose = pose.to(dtype=camera_dirs.dtype)
    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]

    # gaussian disturb
    # disturb_amplitude = pose[:3, 3].norm() / 5
    # pose[:3, 3] += torch.randn_like(pose[:3, 3]) * disturb_amplitude

    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)

    # I_eps = torch.zeros_like(I_min_cov.sum(dim=-3)) + 1e-10
    # p_intersect = torch.pinverse(I_min_cov.sum(dim=-3) + I_eps).matmul(sum_proj)[..., 0]
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    # I_min_cov.sum(dim=-3): torch.Size([1, 1, 3, 3])
    # sum_proj: torch.Size([1, 1, 3, 1])

    # p_intersect = np.linalg.lstsq(I_min_cov.sum(dim=-3).numpy(), sum_proj.numpy(), rcond=None)[0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        return None, None
        ipdb.set_trace()
        assert False
    return p_intersect, r

def compute_optimal_rotation_alignment(A, B):
    """
    Compute optimal R that minimizes: || A - B @ R ||_F

    Args:
        A (torch.Tensor): (N, 3)
        B (torch.Tensor): (N, 3)

    Returns:
        R (torch.tensor): (3, 3)
    """
    # normally with R @ B, this would be A @ B.T
    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    return U @ S_prime @ Vh


def plucker_to_3d(
    rays,
):
    """
    Args:
        rays (Rays): (N, P, 6)
    """
    
    directions = torch.nn.functional.normalize(rays[..., 3:], dim=-1)
    moment = rays[..., :3]

    c = torch.linalg.norm(directions, dim=-1, keepdim=True)
    moment = moment / c

    origins = torch.cross(directions, moment, dim=-1) 

    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)
    
    return camera_centers, directions


def plucker_to_camera(
    rays,
    fovy=49.1, 
):
    """
    Args:
        rays (Rays): (N, P, 6) 
    """
    rays = rays.to(torch.float).cpu()
    directions = torch.nn.functional.normalize(rays[..., 3:], dim=-1)
    moment = rays[..., :3]

    c = torch.linalg.norm(directions, dim=-1, keepdim=True)
    moment = moment / c

    origins = torch.cross(directions, moment, dim=-1) 

    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)
    
    # h = np.sqrt(rays.shape[1])
    # focal_length = [h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))]
    # focal_length = focal_length * rays.shape[0]

    # print(f'[camera_centers] {camera_centers.shape}')

    R = np.zeros((rays.shape[0], 3, 3))
    warp_angle = np.zeros((rays.shape[0], 4, 4))

    _, rays_d_ref = create_ref_ray(directions.shape)

    for i in range(rays.shape[0]):
        R[i] = compute_optimal_rotation_alignment(
            rays_d_ref,
            directions[i],
        )
        warp_angle[i,  3,  3] = 1
        warp_angle[i, :3, :3] = R[i]
        warp_angle[i, :3,  3] = camera_centers[i]
    
    return warp_angle



def create_ref_ray(shape, fovy=49.1, opengl=False):
    ray_o = torch.zeros((1, 3))

    _, P, C = shape # B, 4096, 3

    w = int(math.sqrt(P))
    h = w

    x, y = torch.meshgrid(
        torch.arange(w),
        torch.arange(h),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )
    rays_d = safe_normalize(camera_dirs).unsqueeze(0) #.view(h, w, 3)

    return ray_o.to(torch.float), rays_d.to(torch.float)


def create_ref_camera():
    return np.eye(4)[None]

if __name__ == '__main__':
    x = np.random.rand(2, 4096, 6)
    x = torch.tensor(x)
    plucker_to_camera(x)