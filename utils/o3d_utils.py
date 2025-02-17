import open3d as o3d
from einops import rearrange, repeat
import numpy as np
from PIL import Image
import trimesh
import copy
import math
import os

from utils.ray_utils import create_ref_ray




def draw_cameras(centers, vertexs, ply_path):
    """
        center: [1, 3]
        vertex: [4, 3]
    """
    lines = o3d.geometry.LineSet()

    cnt = 0
    for center, vertex in zip(centers, vertexs):
        cube_points = np.vstack([center, vertex]).astype(np.float64)
        cube_edges  = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
        ]).astype(np.float64)

        lines.points.extend(cube_points)
        lines.lines.extend(cube_edges + cnt * 5)
        cnt += 1

    o3d.io.write_line_set(ply_path, lines)


def create_textured_rectangle(vertices, image_path):
    """
        vertex
    """
    # Load the texture image
    image = Image.open(image_path)
    # image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip to match OpenGL texture coordinates

    vertices = vertices[::-1]
    vertices = vertices[[2, 3, 0, 1]]

    # Create the UV coordinates
    uv = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    # Create the faces for the rectangle
    faces = np.array([
        [0, 1, 2],
        [2, 3, 0]
    ])

    # Create a mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, validate=True)

    # Create a scene to attach the texture
    scene = trimesh.Scene(mesh)

    # Create a material with the texture
    material = trimesh.visual.texture.TextureVisuals(uv=uv, image=image)
    mesh.visual = material

    return mesh


def draw_images(
    vertexs, 
    images,
    obj_path,
):
    assert len(vertexs) == len(images)

    scene = trimesh.Scene()
    for vertex, image in zip(vertexs, images):
        rectangle = create_textured_rectangle(vertex, image)
        scene.add_geometry(copy.deepcopy(rectangle))

    scene.export(obj_path)


def save_images_and_cameras_from_ray(
    rays,
    images,
    root_path=None,
    # names=None,
    camera_scale=1, 
):
    """
        ray: [ray_o, ray_d]
    """
    # assert len(rays) == len(images)
    
    ray_os, ray_ds = [], []
    for ray in rays:
        ray_o, ray_d = ray

        H = int(math.sqrt(ray_d.shape[1]))
        assert H*H == ray_d.shape[1]

        ray_o = ray_o.reshape(-1, 3).detach().cpu().numpy()
        ray_d = ray_d[0].reshape(H, H, 3).detach().cpu().numpy() * camera_scale
        
        ray_o = ray_o[0:1]
        ray_d = [ray_d[-1, -1], ray_d[-1, 0], ray_d[0, 0], ray_d[0, -1]]
        ray_d = np.vstack(ray_d)
        
        ray_os.append(ray_o)
        ray_ds.append(ray_d + ray_o)
    
    draw_cameras(ray_os, ray_ds, os.path.join(root_path, 'camera.ply'))
    draw_images( ray_ds, images, os.path.join(root_path, 'images.obj'))


def save_images_and_cameras_from_cameras(
    c2ws,
    shape, # [B, 4096, 6]
    images,
    root_path=None,
    camera_scale=1, 
    only_camera=False,
    save_npy=True,
    camera_ply_name='camera.ply',
):
    """
        ray: [ray_o, ray_d]
    """
    # assert len(rays) == len(images)
    
    ray_o, ray_d = create_ref_ray(shape)

    H = int(math.sqrt(ray_d.shape[1]))
    assert H*H == ray_d.shape[1]

    ray_o = ray_o.reshape(-1, 3).detach().cpu().numpy()
    ray_d = ray_d.reshape(H, H, 3).detach().cpu().numpy() * camera_scale

    # os.makedirs(root_path, exist_ok=True)
    ray_os, ray_ds = [], []
    for idx, c2w in enumerate(c2ws):
                
        points = [ray_d[-1, -1], ray_d[-1, 0], ray_d[0, 0], ray_d[0, -1]]

        ray_d_now = []
        for i in range(len(points)):
            x = np.ones((4, 1))
            x[:3, 0] = points[i]
            x = c2w @ x
            ray_d_now.append(x[:3].T) 
        ray_d_now = np.vstack(ray_d_now)

        x = np.ones((4, 1))
        x[:3, 0] = ray_o[0]
        x = c2w @ x
        ray_o_now = x[:3].T


        ray_os.append(ray_o_now)
        ray_ds.append(ray_d_now)

        if save_npy:
            np.save(images[idx].replace('.png', '.npy'), c2w)

    draw_cameras(ray_os, ray_ds, os.path.join(root_path, camera_ply_name))

    if not only_camera:
        draw_images(ray_ds, images, os.path.join(root_path, 'images.obj'))