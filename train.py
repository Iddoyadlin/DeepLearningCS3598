import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ProjectDataset
from models.DAE import AE
from utils import to_img

if __name__ == "__main__":

    #Paths
    path_to_img_folder = 'images'
    path_to_save = 'runs'

    ### Model Settings
    lr = 1e-3
    wd = 1e-5
    num_epochs = 100

    model = AE()
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    #Run Settings
    idx_to_save = 0
    save_run_as = osp.join(path_to_save,'run_lr_{}_wd_{}'.format(lr,wd))
    if not osp.exists(save_run_as):
        os.makedirs(save_run_as, exist_ok=True)

    dataset =  ProjectDataset(path_to_img_folder, endswith='.jpg')
    trainloader = DataLoader(dataset, batch_size=1, num_workers=2)

    ##### Training

    loss_history = []

    print('***** Start training *****\n')
    for epoch in range(num_epochs):
        for i, data in tqdm(enumerate(trainloader)):

            noise_img = data['noise_img']
            img = data['img']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, vector = model(noise_img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()

        loss_history.append(loss)
        print('\n epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.data))

        if epoch % 10 == 0:
            img_temp = to_img(outputs)
            img_to_save = np.asarray(img_temp[idx_to_save].permute(1, 2, 0).detach())
            to_save_path = osp.join(save_run_as,'epoch{}.jpg'.format(epoch))
            plt.imsave(to_save_path, np.uint8(img_to_save*255))

    print('***** Done training *****\n')