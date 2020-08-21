from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import PIL.Image as Image

from utils import add_noise_to_img

class ProjectDataset(Dataset):

    def __init__(self, root_dir,device, endswith = '.jpg', cache=True, noise=0.2):
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
        self.device = device

        self.cache = cache
        self.images_cache = [None] * len(self.img_list)
        self.noise = noise

    def get_img_paths(self):
        return self.img_list

    def get_img(self, idx):
        img_name = os.path.join(self.root_dir, self.img_list[idx])
        if self.cache and self.images_cache[idx] is not None:
            image = self.images_cache[idx]
        else:
            image = Image.open(img_name)
            image = self.transform(image)
            image = image.to(self.device)
            if self.cache:
                self.images_cache[idx] = image
        return image

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.img_list[idx])
        image = self.get_img(idx)
        image_noise = add_noise_to_img(image, self.noise)
        sample = {'noise_img': image_noise, 'img' : image, 'path': img_name}
        return sample