import torch
import torch.nn as nn
import torch.nn.functional as F

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