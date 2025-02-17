from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np


def read_image_local(
    image_path, 
    image_size, 
    white_bg=True,
    transform=None
):
    img = Image.open(image_path)#.convert('RGB') #.resize((image_size, image_size))

    if np.array(img).shape[-1] == 4: # png
        if white_bg:
            im_data = np.array(img)
            bg = np.array([1, 1, 1])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            img = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")#.resize((image_size, image_size))            
        else:
            # img = Image.fromarray(im_data).convert("RGB")
            img = img.convert('RGB')
    else:
        img = img.convert('RGB')


    width, height = img.size

    if width > height:
        long_side, short_side = width, height
        left = (width - height) // 2
        upper = 0
        right = left + short_side
        lower = short_side
    else:
        long_side, short_side = height, width
        left = 0
        upper = (height - width) // 2
        right = width
        lower = upper + short_side
    
    img = img.crop((left, upper, right, lower)).resize((image_size, image_size))

    if transform is not None:
        img = transform(img)
    return img