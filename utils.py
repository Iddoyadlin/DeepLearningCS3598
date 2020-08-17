import torch
import numpy as np
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
import shutil

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
    img = img.view(img.size(0),3, 64, 64)
    return img

def add_noise_to_img(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise.to(img.device)
    return noisy_img

def unpack_archive(path):
    shutil.unpack_archive(path)

def decompress_npz_to_images(path_to_file, path_to_decompress):
    """
    Decompress mpi3d_toy.npz into a images in path_to_decompress folder
    :param path_to_file: path to mpi3d_toy.npz
    :param path_to_decompress: Where to decompress?
    """
    data = np.load(path_to_file)['images']
    data = data.reshape([6, 6, 2, 3, 3, 40, 40, 64, 64, 3])
    data_train = data[0, 0, 0, 0, 0, :, :, :, :, :]
    data_train = data_train.reshape(-1, 64, 64, 3)

    if not osp.exists(path_to_decompress):
        os.makedirs(path_to_decompress, exist_ok=True)

    for i, j in tqdm(enumerate(data_train)):
        img = Image.fromarray(j)
        img.save(osp.join(path_to_decompress,'image_{}.jpg'.format(i)))


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
    path_to_save = 'images'
    decompress_npz_to_images(path_to_npz, path_to_save)
