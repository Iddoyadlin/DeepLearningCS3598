import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models.vgg as vgg

from dataset import ProjectDataset
from models.DAE import AE
from models.LossNN import LossNetwork
from utils import to_img, compare_models

if __name__ == "__main__":
    #Paths
    path_to_img_folder = 'images'
    path_to_save = 'runs'

    ### Model Settings
    lr = 1e-2
    wd = 1e-5
    num_epochs = 400
    batch_size = 8
    step_size = 100
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    model = AE().to(device)
    vgg_model = vgg.vgg16(pretrained=True)
    loss_network = LossNetwork(vgg_model).to(device)
    loss_network.eval()

    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    schudler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    #Run Settings
    idx_to_save = 0


    folder = 'run_lr_{}_wd_{}_p_0810'.format(lr,wd)
    save_run_as = osp.join(path_to_save,folder)
    if not osp.exists(save_run_as):
        os.makedirs(save_run_as, exist_ok=True)

    dataset =  ProjectDataset(path_to_img_folder,device)
    trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    ##### Training

    loss_history = []

    print('***** Start training *****\n')
    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, data in tqdm(enumerate(trainloader)):

            noise_img = data['noise_img'].to(device)
            img = data['img'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, vector = model(noise_img)

            with torch.no_grad():
                xc = img.detach()

            features_y = loss_network(outputs)
            features_xc = loss_network(xc)

            with torch.no_grad():
                f_xc_c = features_xc[2].detach()

            loss_c = criterion(features_y[2], f_xc_c)

            loss = 0.3 * criterion(outputs, img) + 0.7 * loss_c
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noise_img.size(0)

        schudler.step()
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        print('\n epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, epoch_loss))

        if epoch % 10 == 0:
            img_temp = to_img(outputs)
            img_to_save = np.asarray(img_temp[idx_to_save].permute(1, 2, 0).detach().cpu())
            to_save_path = osp.join(save_run_as,'epoch{}.jpg'.format(epoch))
            plt.imsave(to_save_path, np.uint8(img_to_save*255))

    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    model.save(os.path.join(save_run_as, 'model_new.pth'))

    new_model = AE()
    new_model.load(osp.join(save_run_as, 'model.pth'))

    print('before saving model was:\n {}'.format(model.state_dict()))
    print('before saving model was:\n {}'.format(new_model.state_dict()))
    compare_models(model, new_model)

    print('***** Done training *****\n')