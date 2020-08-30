import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from config import PATH_TO_SAVE
from dataset import ProjectDataset
from models.DAE import AE

from models.LossNN import LossNetwork
import torchvision.models.vgg as vgg

from utils import to_img, get_device, add_to_zip

RunningInCOLAB = 'google.colab' in str(get_ipython())

if __name__ == "__main__":
    batch_sizes = [4, 8, 16]
    weights = [(1, 0)]
    out_channels = [64, 128, 256]
    noises = [0.1, 0.2, 0.5]
    noise_types = ['normal', 'white']

    run_configurations = []
    for batch_size in batch_sizes:
        for weight in weights:
            for out_channel in out_channels:
                for noise in noises:
                    for noise_type in noise_types:
                        run_configurations.append(dict(
                            lr=1e-2,
                            wd=1e-5,
                            num_epochs=300,
                            batch_size=batch_size,
                            step_size=75,
                            dims=2,
                            criterion_weight=weight[0],
                            loss_network_weight=weight[1],
                            out_channels=out_channel,
                            noise=noise,
                            noise_types=noise_type,
                            IMAGES_PATH='images2d'
                        )
                        )
    print('\n'.join([str(c) for c in run_configurations]))
    print(len(run_configurations))

    for j, run_configuration in enumerate(run_configurations, start=1):
        torch.manual_seed(42)

        lr, wd, num_epochs, batch_size, step_size, dims, criterion_weight, loss_network_weight, out_channels, noise, noise_type, IMAGES_PATH= tuple(run_configuration.values())
        device = get_device()
        model = AE(dims=dims, out_channels=out_channels).to(device)
        vgg_model = vgg.vgg16(pretrained=True)
        loss_network = LossNetwork(vgg_model).to(device)
        loss_network.eval()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

        idx_to_save = 0

        folder = '_'.join(['{}_{}'.format(k, v) for k, v in run_configuration.items()])
        save_run_as = osp.join(PATH_TO_SAVE, folder)
        if osp.exists(save_run_as):
            print('skipping configuration {} already exists\n'.format(save_run_as))
            continue
        if not osp.exists(save_run_as):
            os.makedirs(save_run_as, exist_ok=True)

        dataset = ProjectDataset(IMAGES_PATH, device, noise=noise, noise_type=noise_type)
        trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        ##### Training

        loss_history = []

        print('***** Start training {}/{} configurations *****\n'.format(j, len(run_configurations)))
        for epoch in range(num_epochs):
            run_losses = []
            for i, data in tqdm(enumerate(trainloader)):
                optimizer.zero_grad()

                img = data['img']
                outputs, _ = model(data['noise_img'])
                if loss_network_weight>0:
                    perceptual_loss = model.perceptual_loss(img, outputs, criterion, loss_network, loss_network_weight)
                else:
                    perceptual_loss =0
                criterion_loss = model.loss(img, outputs, criterion)
                loss = criterion_weight * criterion_loss + loss_network_weight* perceptual_loss
                loss.backward()
                optimizer.step()
                run_losses.append(outputs.shape[0] * loss.item())
            if scheduler is not None:
                scheduler.step()
            epoch_loss = sum(run_losses) / len(dataset)
            loss_history.append(epoch_loss)
            print('\n epoch [{}/{}], loss:{:.6f} #config={}'.format(epoch + 1, num_epochs, epoch_loss, j))

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                img_temp = to_img(outputs)
                img_to_save = np.asarray(img_temp[idx_to_save].permute(1, 2, 0).detach().cpu())
                to_save_path = osp.join(save_run_as, 'epoch{}.jpg'.format(epoch))
                plt.imsave(to_save_path, np.uint8(img_to_save * 255))

                loss_path = osp.join(save_run_as, 'losses.txt')
                if epoch == 0 and osp.exists(loss_path):
                    os.remove(loss_path)

                with open(loss_path, 'a+') as f:
                    f.write('{}\n'.format(loss_history[-1]))
        print('***** Done training {}/{} configurations *****\n'.format(j, len(run_configurations)))

        model.save(os.path.join(save_run_as, 'model.pth'))
        with open(os.path.join(save_run_as, 'config.json'), 'w') as f:
            json.dump(run_configuration, f)

        all_files = [os.path.join(save_run_as, 'model.pth'), os.path.join(save_run_as, 'config.json'),
                     osp.join(save_run_as, 'losses.txt')]

        if RunningInCOLAB:
            from google.colab import files
            zip_path = add_to_zip(save_run_as, all_files)