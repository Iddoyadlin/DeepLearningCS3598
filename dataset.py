from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

from utils import add_noise_to_img, to_img

class ProjectDataset(Dataset):

    def __init__(self, root_dir, endswith = 'image_0.jpg'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_list_temp = os.listdir(root_dir)
        self.root_dir = root_dir
        self.img_list = [img_path for img_path in img_list_temp if img_path.endswith(endswith)]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_list[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        image_noise = add_noise_to_img(image)
        sample = {'noise_img': image_noise, 'img' : image}

        return sample

if __name__ == '__main__':

    path = 'images/'
    idx = 0

    test_Dataset_class = ProjectDataset(path)
    test = test_Dataset_class[idx]
    image = test['noise_img'].unsqueeze(0)
    img = to_img(image)[0].permute(1,2,0)
    img = np.asarray(img)
    plt.imshow(img)
    plt.show()