import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=1)

        self.conv3 = nn.Conv2d(32, 16, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(144, 2)

    def forward(self, x):
        x = self.conv1(x)  # [b, 64, 22, 22]
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)  # [b, 64, 11, 11]
        x = self.conv2(x)  # [b, 16, 6, 6]
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)  # [b, 16, 5, 5]
        x = self.conv3(x)  # [b, 2, 2, 2]
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)  # [b, 2, 1, 1]
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, 4096)
        self.deconv1 = nn.ConvTranspose2d(16, 32, 3, stride=2)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(32, 64, 3, stride=2)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(64, 3, 2, stride=1, padding=2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        n_samples = x.size(0)
        x = self.fc1(x)
        x = x.reshape(n_samples, -1, 16, 16)
        x = self.deconv1(x)  # b, 8, 3, 3
        x = self.relu1(x)
        x = self.deconv2(x)  # b, 16, 9, 9
        x = self.relu2(x)
        x = self.deconv3(x)  # b, 32, 16, 16
        x = self.tanh(x)
        return x


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        vector = x.reshape(-1, 2)
        x = self.decoder(vector)
        return x, vector

    def load(self, path):
        dic = torch.load(path)
        e = Encoder()
        d = Decoder()
        e.load_state_dict(dic['encoder'])
        d.load_state_dict(dic['decoder'])
        self.encoder = e
        self.decoder = d

    def save(self, path):
        dic = {'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict()}
        torch.save(dic, path)

    def loss(self, data, criterion):
        noise_img = data['noise_img']
        img = data['img']
        outputs, vector = self(noise_img)
        loss = criterion(outputs, img)
        return loss, outputs

    def perceptual_loss(self, data,  criterion, loss_network, loss_network_weight, criterion_weight):
        noise_img = data['noise_img']
        img = data['img']

        # forward + backward + optimize
        outputs, vector = self(noise_img)

        with torch.no_grad():
            xc = img.detach()

        features_y = loss_network(outputs)
        features_xc = loss_network(xc)

        with torch.no_grad():
            f_xc_c = features_xc[2].detach()
        loss_c = criterion(features_y[2], f_xc_c)
        loss = criterion_weight * criterion(outputs, img) + loss_network_weight * loss_c
        return loss * noise_img.size(0), outputs