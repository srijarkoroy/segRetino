import torch
import torch.nn as nn

class conv_block(nn.Module):

    def __init__(self, in_channel, out_channel):

        """
        This class is used for building a convolutional block. Each convolutional block contains:

        - Conv2d layers (2)

        - BatchNorm2d layers following each Conv2d  

        - ReLU activation after each BatchNorm2d


        Parameters:

        - in_channel: no. of input channels to the convolutional block

        - out_channel: no. of output channels from each convolutional block

        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):

    def __init__(self, in_channel, out_channel):

        """
        This class is used for building a encoder block. Each encoder block contains:

        - Convolutional block (1)

        - MaxPool2d following the convolutional block (1)  


        Parameters:

        - in_channel: no. of input channels to the encoder block

        - out_channel: no. of output channels from each encoder block

        """

        super().__init__()

        self.conv = conv_block(in_channel, out_channel)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):

    def __init__(self, in_channel, out_channel):

        """
        This class is used for building a decoder block. Each decoder block contains:

        - ConvTranspose2d layers (2)

        - Concatenation of upsampled and skip 

        - Convolutional block (2)


        Parameters:

        - in_channel: no. of input channels to the decoder block

        - out_channel: no. of output channels from each decoder block

        """

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_channel+out_channel, out_channel)

    def forward(self, inputs, skip):

        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):

    def __init__(self):

        """
        Main UNET model

        """

        super().__init__()

        """ Encoder """
        self.encoder1 = encoder_block(3, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.decoder1 = decoder_block(1024, 512)
        self.decoder2 = decoder_block(512, 256)
        self.decoder3 = decoder_block(256, 128)
        self.decoder4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ Encoder """
        s1, p1 = self.encoder1(inputs)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        """ Bottleneck """
        bottleneck = self.b(p4)

        """ Decoder """
        d1 = self.decoder1(bottleneck, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        out = self.outputs(d4)

        return out

#noise = torch.randn((2, 3, 512, 512))

#model = UNET()

#print(model)
#print(model(noise).shape)