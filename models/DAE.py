import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, dims, out_channels):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, out_channels, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(out_channels, out_channels//2, 3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=1)

        self.conv3 = nn.Conv2d(out_channels//2, out_channels//4, 3, stride=2)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(144 * (out_channels//64), dims) # hack to figure out linear size...

    def forward(self, x):
        x = self.conv1(x)  # [b, 64, 32, 32]

        x = self.relu1(x)
        x = self.maxpool1(x)  # [b, 64, 16, 16]
        x = self.conv2(x)  # [b, 32, 8, 8]

        x = self.relu2(x)
        x = self.maxpool2(x)  # [b, 32, 7, 7]
        x = self.conv3(x)  # [b, 16, 3, 3]

        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)  # [b, 2, 1, 1]
        return x


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(dims, 4096)
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

    def __init__(self, dims, out_channels):
        super(AE, self).__init__()
        self.encoder = Encoder(dims, out_channels)
        self.decoder = Decoder(dims)

    def forward(self, x):
        x = self.encoder(x)
        vector = x.reshape(-1, x.shape[1])
        x = self.decoder(vector)
        return x, vector

    def load(self, path, dims, device, out_channels):
        dic = torch.load(path, map_location=torch.device(device))
        e = Encoder(dims, out_channels=out_channels)
        d = Decoder(dims)
        e.load_state_dict(dic['encoder'])
        d.load_state_dict(dic['decoder'])
        self.encoder = e
        self.decoder = d

    def save(self, path):
        dic = {'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict()}
        torch.save(dic, path)

    def loss(self, img, outputs, criterion):
        loss = criterion(outputs, img)
        return loss

    def perceptual_loss(self, img, outputs, criterion, loss_network, loss_network_weight):
        with torch.no_grad():
            xc = img.detach()
        features_y = loss_network(outputs)
        features_xc = loss_network(xc)
        with torch.no_grad():
            f_xc_c = features_xc[2].detach()
        loss_c = criterion(features_y[2], f_xc_c)
        loss = loss_network_weight * loss_c
        return loss