from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=1)

        self.conv3 = nn.Conv2d(8, 2, 3, stride=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = self.conv1(x)           # [b, 64, 22, 22]
        x = self.relu1(x)
        x = self.maxpool1(x)        # [b, 64, 11, 11]
        x = self.conv2(x)           # [b, 16, 6, 6]
        x = self.relu2(x)
        x = self.maxpool2(x)        # [b, 16, 5, 5]
        x = self.conv3(x)           # [b, 2, 2, 2]
        x = self.relu3(x)
        x = self.maxpool3(x)        # [b, 2, 1, 1]
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(2, 32, 3, stride=2)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(16, 8, 2, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(4, 3, 2, stride=2, padding=0)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = x.reshape(-1,2,1,1) #b, 2, 1, 1
        x = self.deconv1(x)     #b, 8, 3, 3
        x = self.relu1(x)
        x = self.deconv2(x)     #b, 16, 9, 9
        x = self.relu2(x)
        x = self.deconv3(x)     # b, 32, 16, 16
        x = self.relu3(x)
        x = self.deconv4(x)
        x = self.relu4(x)
        x = self.deconv5(x)
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