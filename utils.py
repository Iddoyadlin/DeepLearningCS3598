import zipfile

import torch
import numpy as np
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
import shutil
from enum import Enum


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_img(img):
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    img = img.view(img.size(0), 3, 64, 64)
    return img


def add_to_zip(destination, sources):
  filepath = destination +'.zip'
  with zipfile.ZipFile(filepath, 'a') as zipf:
    for source_path in sources:
      destination = source_path.split('/')[-1]
      zipf.write(source_path, destination)
  return filepath

def add_noise_to_img(img, noise_param=0.2, noise_type='normal'):
    if noise_type == 'normal':
        noise = torch.randn(img.size()) * noise_param
        return img + noise.to(img.device)
    else:
        img = img.clone()
        noise = torch.rand(img.size())
        img[noise < noise_param] = 0
        return img


def unpack_archive(path):
    shutil.unpack_archive(path)


def get_data_indices(data, dim, indices):
    if indices == 'all':
        return np.arange(data.shape[dim])
    else:
        return indices


class ObjectColor(Enum):
    WHITE = 0
    GREEN = 1
    RED = 2
    BLUE = 3
    BROWN = 4
    OLIVE = 5
    ALL = [0, 1, 2, 3, 4, 5]
    ENUM_NAME = 'object_color'


class ObjectShape(Enum):
    CONE = 0
    CUBE = 1
    CYLINDER = 2
    HEXAGONAL = 3
    PYRAMID = 4
    SPHERE = 5
    ALL = [0, 1, 2, 3, 4, 5]
    ENUM_NAME = 'object_shape'


class ObjectSize(Enum):
    SMALL = 0
    LARGE = 1
    ALL = [0, 1]
    ENUM_NAME = 'object_size'


class CameraHeight(Enum):
    TOP = 0
    CENTER = 1
    BOTTOM = 2
    ALL = [0, 1, 2]
    ENUM_NAME = 'camera_height'


class BackgroundColor(Enum):
    PURPLE = 0
    SEA_GREEN = 1
    SALMON = 2
    ALL = [0, 1, 2]
    ENUM_NAME = 'background_color'


def decompress_npz_to_images(path_to_file, object_color,
                             object_shape, object_size,
                             camera_height, background_color):
    """
    Decompress mpi3d_toy.npz into a images in path_to_decompress folder
    :param path_to_file: path to mpi3d_toy.npz
    :param path_to_decompress: Where to decompress?
    """
    params = [object_color, object_shape, object_size, camera_height, background_color]
    assert len([p for p in params if p.name == 'ALL']) <=1, 'please supply only one extra dimension'


    data = np.load(path_to_file)['images']
    data = data.reshape([6, 6, 2, 3, 3, 40, 40, 64, 64, 3])

    data_train = data[object_color.value, object_shape.value, object_size.value, camera_height.value,
                 background_color.value, :, :, :, :, :]
    data_train = data_train.reshape(-1, 64, 64, 3)



    params_all = ''.join([p.__objclass__.ENUM_NAME.value for p in params if p.name == 'ALL'])
    if params_all:
        path_to_decompress = 'images3d_' + params_all
    else:
        path_to_decompress = 'images2d'
    print('decompressing {} images to {}'.format(data_train.shape[0], path_to_decompress))

    if not osp.exists(path_to_decompress):
        os.makedirs(path_to_decompress, exist_ok=True)

    for i, j in tqdm(enumerate(data_train)):
        img = Image.fromarray(j)
        img.save(osp.join(path_to_decompress, 'image_{}.jpg'.format(i)))


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1].cpu(), key_item_2[1].cpu()):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


if __name__ == '__main__':
    path_to_npz = 'mpi3d_toy.npz'
    decompress_npz_to_images(path_to_npz, object_color=ObjectColor.WHITE,
                             object_shape=ObjectShape.CONE, object_size=ObjectSize.LARGE,
                             camera_height=CameraHeight.TOP, background_color=BackgroundColor.PURPLE)
