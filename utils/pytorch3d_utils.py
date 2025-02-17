# import pytorch3d
import gzip
import json
import os.path as osp
import random
import time
import os
# import ipdb  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
# from pytorch3d.renderer import PerspectiveCameras
# from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
# from ray_diffusion.utils.ceph_utils import PetrelBackend
import cv2



def flip_rotation_axes(rotation_matrix, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a 3x3 rotation matrix.

    Args:
    rotation_matrix (np.array): The original 3x3 rotation matrix.
    flip_x (bool): Whether to flip the X-axis.
    flip_y (bool): Whether to flip the Y-axis.
    flip_z (bool): Whether to flip the Z-axis.

    Returns:
    np.array: The rotation matrix after flipping the specified axes.
    """
    flipped_matrix = rotation_matrix.copy()

    if flip_x:
        flipped_matrix[1:3, :] = -flipped_matrix[1:3, :]

    if flip_y:
        flipped_matrix[[0, 2], :] = -flipped_matrix[[0, 2], :]

    if flip_z:
        flipped_matrix[:, [0, 1]] = -flipped_matrix[:, [0, 1]]

    return flipped_matrix

def flip_translation_vector(translation_vector, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a translation vector.

    Args:
    translation_vector (np.array): The original translation vector.
    flip_x (bool): Whether to flip along the X-axis.
    flip_y (bool): Whether to flip along the Y-axis.
    flip_z (bool): Whether to flip along the Z-axis.

    Returns:
    np.array: The translation vector after flipping the specified axes.
    """
    flipped_vector = translation_vector.copy()

    if flip_x:
        flipped_vector[0] = -flipped_vector[0]

    if flip_y:
        flipped_vector[1] = -flipped_vector[1]

    if flip_z:
        flipped_vector[2] = -flipped_vector[2]

    return flipped_vector


def reflip_rotation_axes(rotation_matrix, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a 3x3 rotation matrix.

    Args:
    rotation_matrix (np.array): The original 3x3 rotation matrix.
    flip_x (bool): Whether to flip the X-axis.
    flip_y (bool): Whether to flip the Y-axis.
    flip_z (bool): Whether to flip the Z-axis.

    Returns:
    np.array: The rotation matrix after flipping the specified axes.
    """
    flipped_matrix = rotation_matrix.copy()

    if flip_z:
        flipped_matrix[:, [0, 1]] = -flipped_matrix[:, [0, 1]]    

    if flip_y:
        flipped_matrix[[0, 2], :] = -flipped_matrix[[0, 2], :]

    if flip_x:
        flipped_matrix[1:3, :] = -flipped_matrix[1:3, :]

    

    return flipped_matrix

def reflip_translation_vector(translation_vector, flip_x=False, flip_y=False, flip_z=False):
    """
    Flip the specified axes of a translation vector.

    Args:
    translation_vector (np.array): The original translation vector.
    flip_x (bool): Whether to flip along the X-axis.
    flip_y (bool): Whether to flip along the Y-axis.
    flip_z (bool): Whether to flip along the Z-axis.

    Returns:
    np.array: The translation vector after flipping the specified axes.
    """
    flipped_vector = translation_vector.copy()

    if flip_z:
        flipped_vector[2] = -flipped_vector[2]

    if flip_y:
        flipped_vector[1] = -flipped_vector[1]
    
    if flip_x:
        flipped_vector[0] = -flipped_vector[0]

    return flipped_vector



# def read_npy(
#     bucket_path, client,
#     obj_name, idx
# ):
#     npz_path = os.path.join(
#         bucket_path, obj_name, 'images', f'frame{str(idx).zfill(6)}.npz'
#     )
#     x = client.get_npy(npz_path)
#     c2w = x['camera_pose'] # opencv = colmap c2w

#     # c2w[:2] *= -1
#     # c2w[:, :2] *= -1

#     w2c = np.linalg.inv(c2w)
#     # w2c = c2w
    
#     # R = w2c[:3, :3]
#     # T = w2c[:3, 3]

#     # Extract rotation matrix 
#     R_colmap = w2c[:3, :3]

#     # Converting COLMAP to PyTorch3D format
#     R_pt3d = flip_rotation_axes(R_colmap, flip_x=True, flip_y=True, flip_z=False)
#     R_pt3d = R_pt3d.T

#     # Extract translation vector
#     T_colmap = w2c[:3, 3]

#     # Converting COLMAP to PyTorch3D format
#     T_pt3d = flip_translation_vector(T_colmap, flip_x=True, flip_y=True, flip_z=False)

#     intr = x['camera_intrinsics']
#     # return R, T, intr
#     return R_pt3d, T_pt3d, intr


def w2c_colmap_to_pytorch(w2c):
    # Extract rotation matrix 
    R_colmap = w2c[:3, :3]

    # Converting COLMAP to PyTorch3D format
    R_pt3d = flip_rotation_axes(R_colmap, flip_x=True, flip_y=True, flip_z=False)
    R_pt3d = R_pt3d.T

    # Extract translation vector
    T_colmap = w2c[:3, 3]

    # Converting COLMAP to PyTorch3D format
    T_pt3d = flip_translation_vector(T_colmap, flip_x=True, flip_y=True, flip_z=False)

    # intr = x['camera_intrinsics']
    # return R, T, intr
    return R_pt3d, T_pt3d#, intr

def w2c_pytorch_to_colmap(w2c):
    # Extract rotation matrix 
    R_colmap = w2c[:3, :3]

    # Converting COLMAP to PyTorch3D format
    R_colmap = R_colmap.T
    R_pt3d = reflip_rotation_axes(R_colmap, flip_x=True, flip_y=True, flip_z=False)
    

    # Extract translation vector
    T_colmap = w2c[:3, 3]

    # Converting COLMAP to PyTorch3D format
    T_pt3d = reflip_translation_vector(T_colmap, flip_x=True, flip_y=True, flip_z=False)

    # intr = x['camera_intrinsics']
    # return R, T, intr
    return R_pt3d, T_pt3d#, intr