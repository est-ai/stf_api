import torch
from torch import nn
#from .conv import (Conv2d, Conv2dTranspose, ResidualConv2d)

bias = True 
inplace = False 
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, act=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(cout, momentum=0.8, eps=0.001),
        )
        self.relu = nn.ReLU(inplace=inplace) if act else None

    def forward(self, x):
        x1 = self.block(x)
        if self.relu: 
            return self.relu(x1)
        else:
            return x1

        
class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, act=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(cout, momentum=0.8, eps=0.001)
        )
        self.relu = nn.ReLU(inplace=inplace) if act else None

    def forward(self, x):
        x1 = self.conv_block(x)
        if self.relu:
            return self.relu(x1)
        else:
            return x1
        
        
class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv2d(c, c, 3, 1, 1)
        self.conv2 = Conv2d(c, c, 3, 1, 1)
        self.relu = nn.ReLU(inplace=inplace)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x2 + x
        x3 = self.relu(x3)
        return x3

class Encoder(nn.Module):
    def __init__(self, shape):
        super().__init__()

        c, h, w = shape

        self.id_map  = Conv2d(c, 32, 7, 1, 3)

        self.conv1 = Conv2d(32, 64, 5, 2, 2)
        self.residual11 = Residual(64)
        self.residual12 = Residual(64)

        self.conv2 = Conv2d(64, 128, 3, 2, 1)
        self.residual21 = Residual(128)
        self.residual22 = Residual(128)
        self.residual23 = Residual(128)
        
        self.conv3 = Conv2d(128, 256, 3, 2, 1)
        self.residual31 = Residual(256)
        self.residual32 = Residual(256)

        self.conv4 = Conv2d(256, 512, 3, 2, 1)
        self.residual41 = Residual(512)
        self.residual42 = Residual(512)

        self.conv5 = Conv2d(512, 512, 3, 2, 1)

        self.conv6 = Conv2d(512, 512, 3, 1, 0)

        kh, kw = ((h + 31) // 32 - 2), ((w + 31) // 32 - 2)
        #kh, kw = 2 * (kh // 2) + 1,  2 * (kw // 2) + 1
        self.conv7 = Conv2d(512, 512, (kh, kw), 1, 0)

    def forward(self, x):
        id_map = self.id_map(x)     # 32: 256, 108,  96

        ft10 = self.conv1(id_map)    # 64: 128,  54,  48
        ft11 = self.residual11(ft10)
        ft12 = self.residual12(ft11)

        ft20 = self.conv2(ft12)      # 128: 64,  27,  24
        ft21 = self.residual21(ft20)
        ft22 = self.residual22(ft21)
        ft23 = self.residual23(ft22)

        ft30 = self.conv3(ft23)      # 256: 32,  14,  12
        ft31 = self.residual31(ft30)
        ft32 = self.residual32(ft31)

        ft40 = self.conv4(ft32)       # 512: 16,   7,   6
        ft41 = self.residual41(ft40)
        ft42 = self.residual42(ft41)

        ft50 = self.conv5(ft42)       # 512:  8,   4,   3
        ft60 = self.conv6(ft50)       # 512:  6,   2,   1
        ft70 = self.conv7(ft60)       # 512:  1,   1,   1

        return [id_map, ft12, ft23, ft32, ft42, ft50, ft60, ft70]

class EncoderAudio(nn.Module):
    def __init__(self, shape):
        super().__init__()

        c, h, w = shape # 1, 96, 108

        self.conv1 = Conv2d(c, 32, 3, 1, 1)
        self.residual11 = Residual(32)
        self.residual12 = Residual(32)

        self.conv2 = Conv2d(32, 64, 3, 3, 1)
        self.residual21 = Residual(64)
        self.residual22 = Residual(64)
        
        #sh, sw = (h + 26) // 27, (w + 26) // 27
        #self.conv3 = Conv2d(64, 128, (5, 5), (sh, sw), (sh//2, sw//2))
        #k = (w+26)//27 # w=108 => k=4
        self.conv3 = Conv2d(64, 128, 3, (3, 3), 1)
        self.residual31 = Residual(128)
        self.residual32 = Residual(128)

        self.conv4 = Conv2d(128, 256, 3, 3, 1)
        self.residual41 = Residual(256)
        self.residual42 = Residual(256)

        self.conv5 = Conv2d(256, 512, 4, 1, 0)

        self.conv6 = Conv2d(512, 512, 1, 1, 0)


    def forward(self, x):

        ft10 = self.conv1(x)          # 96x108    
        ft11 = self.residual11(ft10) 
        ft12 = self.residual12(ft11)

        ft20 = self.conv2(ft12)       # 32x36      
        ft21 = self.residual21(ft20)
        ft22 = self.residual22(ft21)

        ft30 = self.conv3(ft22)       # 11x9 | 11x12  
        ft31 = self.residual31(ft30)
        ft32 = self.residual32(ft31)

        ft40 = self.conv4(ft32)       # 4x3 | 4x4       
        ft41 = self.residual41(ft40)
        ft42 = self.residual42(ft41)

        ft50 = self.conv5(ft42)  # 1x1 | 1, 1     
        ft60 = self.conv6(ft50)  # 1x1     

        return ft60

class Decoder(nn.Module):
    def __init__(self, shape):
        super().__init__()

        c, h, w = shape
        kh, kw = (h + 31) // 32, (w + 31) // 32
        self.convt1 = Conv2dTranspose(1024, 512, (kh, kw), (kh, kw), 0)

        self.convt2 = Conv2dTranspose(1024, 512, 3, 2, 1, 1)
        self.residual21 = Residual(512)
        self.residual22 = Residual(512)

        self.convt3 = Conv2dTranspose(1024, 256, 3, 2, 1, 1)
        self.residual31 = Residual(256)
        self.residual32 = Residual(256)

        self.convt4 = Conv2dTranspose(512, 128, 3, 2, 1, 1)
        self.residual41 = Residual(128)
        self.residual42 = Residual(128)

        self.convt5 = Conv2dTranspose(256, 64, 3, 2, 1, 1)
        self.residual51 = Residual(64)
        self.residual52 = Residual(64)

        self.convt6 = Conv2dTranspose(128, 32, 3, 2, 1, 1)

        self.conv7 = Conv2d(64, 16, 3, 1, 1)
        self.conv8 = Conv2d(16, 16, 3, 1, 1)
        self.conv9 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.tanh= nn.Tanh()

    def forward(self, img_ft, audio_ft):
        x = torch.cat([img_ft[-1], audio_ft], dim=1)  # (B, 1024, 1, 1)
                                #          256,  96,
            
        x = self.convt1(x)      # (B, 512:  8,  3)
        
        x = torch.cat([img_ft[5], x], dim=1)
        
        x = self.convt2(x)      # (B, 512: 16,  6)
        x = self.residual21(x)
        x = self.residual22(x)
        x = torch.cat([img_ft[4], x], dim=1)
        
        x = self.convt3(x)      # (B, 256: 32, 12)
        x = self.residual31(x)
        x = self.residual32(x)
        x = torch.cat([img_ft[3], x], dim=1)
        
        x = self.convt4(x)      # (B, 128: 64, 24)
        x = self.residual41(x)
        x = self.residual42(x)
        x = torch.cat([img_ft[2], x], dim=1)
        
        x = self.convt5(x)      # (B, 64: 128, 48)
        x = self.residual51(x)
        x = self.residual52(x)
        x = torch.cat([img_ft[1], x], dim=1)
        
        x = self.convt6(x)      # (B, 32: 256, 96)
        x = torch.cat([img_ft[0], x], dim=1)
        x = self.conv7(x)       # (B, 16: 256, 96)
        x = self.conv8(x)       # (B, 16: 256, 96)
        
        x = self.conv9(x)       # (B,  3: 256, 96)
        x = self.tanh(x)
        
        return x


class Speech2Face(nn.Module):

    def __init__(self, img_num, img_shape, audio_shape):
        super().__init__()

        self.speech_encoder = EncoderAudio(audio_shape)

        c, h, w = img_shape
        c = c * img_num
        self.face_encoder  = Encoder((c, h, w))
        self.face_decoder  = Decoder(img_shape)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if bias == False:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                        
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, audio):

        img   = self.face_encoder(img)
        audio = self.speech_encoder(audio)

        img   = self.face_decoder(img, audio)
        return img