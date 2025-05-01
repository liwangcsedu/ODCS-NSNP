import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SRS_Operator import SRS_Operator

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SDN_Units(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDN_Units, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sconv1 = SeparableConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels + in_channels)
        self.sconv2 = SeparableConv2d(in_channels + out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*2 + in_channels)
        self.sconv3 = SeparableConv2d(in_channels + out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(in_channels + out_channels*3,out_channels,1,1,bias=False)

    def forward(self, x):
        # Activate conv 3*3, transform to ConvSNP 3*3 convolution
        x = self.bn1(x)
        x1 = self.relu(x)
        out1 = self.sconv1(x1)
        # Activate conv 3*3, transform to ConvSNP 3*3 convolution
        inp2 = torch.cat((x,out1),1)
        x2 = self.bn2(inp2)
        x2 = self.relu(x2)
        out2 = self.sconv2(x2)
        # Activate conv 3*3, transform to ConvSNP 3*3 convolution
        inp3 = torch.cat((inp2,out2),1)
        x3 = self.bn3(inp3)
        x3 = self.relu(x3)
        out3 = self.sconv3(x3)

        out = torch.cat((inp3,out3),1)
        # Activate conv 1*1, transform to ConvSNP 1*1 convolution
        out = self.relu(out)
        out = self.conv1_1(out)

        return out

class ODCS_NSNP(nn.Module):

    def __init__(self, in_channel=3,out_channel=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(ODCS_NSNP, self).__init__()
        # Sub-block input
        self.subsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.subsample_conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.subsample_conv2 = nn.Conv2d(3, 128, 3, 1, padding=1, bias=False)
        self.subsample_conv3 = nn.Conv2d(3, 256, 3, 1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        # Replace MaxPool2d with SRS_Operator for downsampling
        self.pool0 = SRS_Operator(scale_factor=0.5, in_channels=32, out_channels=32)

        self.block1 = SDN_Units(96, 64)
        self.block1_2 = SDN_Units(64,64)
        self.block1_3 = SDN_Units(64,64)

        # Replace MaxPool2d with SRS_Operator for downsampling
        self.pool1 = SRS_Operator(scale_factor=0.5, in_channels=64, out_channels=64)

        self.block2 = SDN_Units(192, 128)
        self.block2_2 = SDN_Units(128, 128)
        self.block2_3 = SDN_Units(128, 128)
        self.block2_4 = SDN_Units(128, 128)

        # Replace MaxPool2d with SRS_Operator for downsampling
        self.pool2 = SRS_Operator(scale_factor=0.5, in_channels=128, out_channels=128)

        self.block3 = SDN_Units(384, 256)
        self.block3_2 = SDN_Units(256, 256)
        self.block3_3 = SDN_Units(256, 256)
        self.block3_4 = SDN_Units(256, 256)
        self.block3_5 = SDN_Units(256, 256)

        # Replace MaxPool2d with SRS_Operator for downsampling
        self.pool3 = SRS_Operator(scale_factor=0.5, in_channels=256, out_channels=256)

        self.block4 = SDN_Units(256, 512)
        self.block4_2 = SDN_Units(512, 512)
        self.block4_3 = SDN_Units(512, 512)
        self.block4_4 = SDN_Units(512, 512)
        self.block4_5 = SDN_Units(512, 512)

        # Replace TransitionUp with SRS_Operator for upsampling
        self.upcon4 = SRS_Operator(scale_factor=2, in_channels=512, out_channels=256)

        self.block5 = SDN_Units(512, 256)
        self.block5_2 = SDN_Units(256, 256)
        self.block5_3 = SDN_Units(256, 256)
        self.block5_4 = SDN_Units(256, 256)
        self.block5_5 = SDN_Units(256, 256)

        # Replace TransitionUp with SRS_Operator for upsampling
        self.upcon5 = SRS_Operator(scale_factor=2, in_channels=256, out_channels=128)

        self.block6 = SDN_Units(256, 128)
        self.block6_2 = SDN_Units(128, 128)
        self.block6_3 = SDN_Units(128, 128)
        self.block6_4 = SDN_Units(128, 128)

        # Replace TransitionUp with SRS_Operator for upsampling
        self.upcon6 = SRS_Operator(scale_factor=2, in_channels=128, out_channels=64)

        self.block7 = SDN_Units(128, 64)
        self.block7_2 = SDN_Units(64, 64)
        self.block7_3 = SDN_Units(64, 64)

        # Replace TransitionUp with SRS_Operator for upsampling
        self.upcon7 = SRS_Operator(scale_factor=2, in_channels=64, out_channels=32)

        self.bn8 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 32, 3, 1, padding=1, bias=False)

        self.bn9 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, out_channel, 3, 1, padding=1, bias=False)


    def forward(self, inp):

        x = self.bn1(inp)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x0 = self.conv2(x)

        sub1 = self.subsample(inp)
        # Apply ConvSNP 3*3 convolution activation to subsample layer
        sub1 = self.relu(sub1)
        s1= self.subsample_conv1(sub1)
        x = self.pool0(x0)

        x = torch.cat((x,s1),1)
        x1_1 = self.block1(x)   # 96*120*120->64*120*120
        x1_2 = self.block1_2(x1_1)
        x1_3 = self.block1_3(x1_2)
        x = self.pool1(x1_3)

        sub2 = self.subsample(sub1)
        # Apply ConvSNP 3*3 convolution activation to subsample layer
        sub2 = self.relu(sub2)
        s2 = self.subsample_conv2(sub2)

        x = torch.cat((x,s2),1)
        x2_1 = self.block2(x)   # 192*60*60->128*60*60
        x2_2 = self.block2_2(x2_1)
        x2_3 = self.block2_3(x2_2)
        x2_4 = self.block2_4(x2_3)

        x= self.pool2(x2_4)

        sub3 = self.subsample(sub2)
        # Apply ConvSNP 3*3 convolution activation to subsample layer
        sub3 = self.relu(sub3)
        s3 = self.subsample_conv3(sub3)

        x = torch.cat((x,s3),1)
        x3_1 = self.block3(x)   # 384*30*30->256*30*30
        x3_2 = self.block3_2(x3_1)
        x3_3 = self.block3_3(x3_2)
        x3_4 = self.block3_4(x3_3)
        x3_5 = self.block3_5(x3_4)


        x = self.pool3(x3_5)

        x4_1 = self.block4(x)   # 512
        x4_2 = self.block4_2(x4_1)
        x4_3 = self.block4_3(x4_2)
        x4_4 = self.block4_4(x4_3)
        x4_5 = self.block4_5(x4_4)


        up4 = self.upcon4(x4_5)

        x = torch.cat((x3_5,up4),1)
        x5_1 = self.block5(x)
        x5_2 = self.block5_2(x5_1)
        x5_3 = self.block5_3(x5_2)
        x5_4 = self.block5_4(x5_3)
        x5_5 = self.block5_5(x5_4)


        up5 = self.upcon5(x5_5)

        x = torch.cat((x2_4,up5),1)
        x6_1 = self.block6(x)
        x6_2 = self.block6_2(x6_1)
        x6_3 = self.block6_3(x6_2)
        x6_4 = self.block6_4(x6_3)

        up6 = self.upcon6(x6_4)
        x = torch.cat((x1_3,up6),1)
        x7_1 = self.block7(x)
        x7_2 = self.block7_2(x7_1)
        x7_3 = self.block7_3(x7_2)

        up7 = self.upcon7(x7_3)

        x = torch.cat((x0,up7),1)

        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv8(x)

        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv9(x)
        return x
