import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, max_pool_flag=True):
        super(DownBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.maxpool = nn.MaxPool2d(3,stride=2,padding=1)
        self.max_pool_flag = max_pool_flag
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

        self.conv_init(self.conv1)
        self.conv_init(self.conv2)
        
    def forward(self,img):
        tmp_img = img
        if self.max_pool_flag:
            tmp_img = self.maxpool(tmp_img)
        tmp_img = self.relu(self.bn(self.conv1(tmp_img)))
        tmp_img = self.relu(self.bn(self.conv2(tmp_img)))
        return tmp_img
    def conv_init(self,layer,lower=-1,upper=1):
        kaiming_uniform_(layer.weight)
        # kaiming_uniform_(layer.bias)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pre_ch):
        super(UpBlock,self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.conv2 = nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.conv3 = nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

        self.conv_init(self.conv1)
        self.conv_init(self.conv2)
        self.conv_init(self.conv3)
    
    def forward(self, img, pre_feat):
        tmp_img = self.upsample(img)
        tmp_img = self.relu(self.bn(self.conv1(tmp_img)))
        tmp_img = torch.cat((tmp_img,pre_feat),dim=1)
        tmp_img = self.relu(self.bn(self.conv2(tmp_img)))
        tmp_img = self.relu(self.bn(self.conv3(tmp_img)))
        return tmp_img

    def conv_init(self,layer,lower=-1,upper=1):
        kaiming_uniform_(layer.weight)
        # kaiming_uniform_(layer.bias)
        
        
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.base_ch = 6
        self.d_block1 = DownBlock(3,self.base_ch,False)
        self.d_block2 = DownBlock(self.base_ch,self.base_ch*2,True)
        self.d_block3 = DownBlock(self.base_ch*2,self.base_ch*4,True)
        self.d_block4 = DownBlock(self.base_ch*4,self.base_ch*8,True)
        self.d_block5 = DownBlock(self.base_ch*8,self.base_ch*16,True)
        
        
        self.u_block1 = UpBlock(self.base_ch*16,self.base_ch*8,self.base_ch*8)
        self.u_block2 = UpBlock(self.base_ch*8,self.base_ch*4,self.base_ch*4)
        self.u_block3 = UpBlock(self.base_ch*4,self.base_ch*2,self.base_ch*2)
        self.u_block4 = UpBlock(self.base_ch*2,self.base_ch*1,self.base_ch*1)
        
        self.last_conv = nn.Conv2d(self.base_ch,2,3,padding=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,img_data):
        img_down1 = self.d_block1(img_data)
        img_down2 = self.d_block2(img_down1)
        img_down3 = self.d_block3(img_down2)
        img_down4 = self.d_block4(img_down3)
        img_down5 = self.d_block5(img_down4)
        
        # print(img_down5.size())
        # print(img_down4.size())
        img_up1 = self.u_block1(img_down5,img_down4)
#         print(img_up1.size())
#         print(img_down3.size())
        img_up2 = self.u_block2(img_up1,img_down3)
        img_up3 = self.u_block3(img_up2,img_down2)
        img_up4 = self.u_block4(img_up3,img_down1)
        
        img_last = self.last_conv(img_up4)
        img_last = self.log_softmax(img_last)
        
        return img_last