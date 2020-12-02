import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DNN4SimBase(nn.Module):
    def training_step(self, batch, loss_func=F.smooth_l1_loss):
        images, labels = batch 
        out = self(images)            # Generate predictions
        loss = loss_func(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch, acc_func, loss_func=F.smooth_l1_loss):
        images, labels = batch 
        out = self(images)              # Generate predictions
        loss = loss_func(out, labels)   # Calculate loss
        acc = acc_func(out, labels)     # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

### RCAN ###
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(DNN4SimBase):
    def __init__(self, conv=default_conv, **args):
        super(RCAN, self).__init__()
        
        n_resgroups = args.get('n_resgroups', 10)
        n_resblocks = args.get('n_resblocks', 20)
        n_feats = args.get('n_feats', 64)
        kernel_size = 3
        reduction = args.get('reduction', 16)
        scale = 1
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(1, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x

### U-Net ###
        
class CUNet(DNN4SimBase):
    def __init__(self):
        super(CUNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU())
        self.down2 = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU())
        self.down3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU())
        self.down4 = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU())
        self.downF = nn.Sequential(nn.MaxPool2d(2, 2),
                                   nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        self.up1  =  nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.up2  =  nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.up3  =  nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.up4  =  nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1,1)),
                                   nn.ReLU())
        self.upF  =  nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1))
        
        
    def forward(self, x):
        return self.upF(self.up4(torch.cat((self.down1(x),self.up3(torch.cat((self.down2(self.down1(x)),self.up2(torch.cat((self.down3(self.down2(self.down1(x))),self.up1(torch.cat((self.down4(self.down3(self.down2(self.down1(x)))),self.downF(self.down4(self.down3(self.down2(self.down1(x)))))), dim=1))), dim=1))), dim=1))), dim=1)))                 