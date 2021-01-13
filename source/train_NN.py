# Usages:
# python train_nn.py features_path labels_path out_folder batch_size epochs (pretrained_path)
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import *
from laploss import LapLoss
import json
import os
import errno
from pytorch_msssim import ms_ssim, MS_SSIM
import pytorch_ssim
import math

def get_default_device(id=0):
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{id}')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
        
def get_train_val(features, labels, train_ratio=0.8, batch_size=10):
    if features.shape[0] != labels.shape[0]:
        raise ValueError('Features and Labels are not of the same size')
    if len(features.shape) != 3:
        raise ValueError('Features and Labels should be 3-dimensional')

    length = features.shape[0]
    
    if length % batch_size != 0:
        features = features[:-(length % batch_size)]
        labels = labels[:-(length % batch_size)]
    
    features = np.reshape(features, (-1, batch_size, 1, features.shape[1], features.shape[2]))
    labels = np.reshape(labels, (-1, batch_size, 1, labels.shape[1], labels.shape[2]))
    print(f'Data size: {len(features), features[0].shape[0], features[0].shape[1], features[0].shape[2], features[0].shape[3]}')

    n = int(train_ratio*features.shape[0])
    
    train_set = [(torch.FloatTensor(features[i]), torch.FloatTensor(labels[i])) for i in range(n)]
    val_set = [(torch.FloatTensor(features[i]), torch.FloatTensor(labels[i])) for i in range(n, features.shape[0])]
    print(f'Train length: {len(train_set)}\nValidation length: {len(val_set)}')
    return train_set, val_set

def load_dataset(feat_path, lab_path, train_ratio=0.8, batch_size=10, gpu_id=0):
    # Load data
    features = []
    with open(feat_path, 'rb') as f:
        features = np.load(f)
    
    labels = []
    with open(lab_path, 'rb') as f:
        labels = np.load(f)
    
    # Split into training and validation set
    train_set, val_set = get_train_val(features, labels, train_ratio, batch_size)
    
    # Move data to GPU
    train_loader = DeviceDataLoader(train_set, get_default_device(gpu_id))
    val_loader = DeviceDataLoader(val_set, get_default_device(gpu_id))
    
    return train_loader, val_loader

# Custom loss function combining Smooth L1 Loss and SSIM
def custom_loss(output, target):
    ssim_l = pytorch_ssim.SSIM()
    sl1l = F.smooth_l1_loss
    return 0.16 * sl1l(output, target) + 0.84 * (1 - ssim_l(output, target))
    
def ssim_loss(output, target):
    ssim_l = pytorch_ssim.SSIM()
    return (1 - ssim_l(output, target))
    
# Custom loss function combining Laplacian Pyramid Loss and L1
def lap_loss_mix(x, y):
    lap_loss = LapLoss(max_levels=1, channels=1)
    l1_loss = F.smooth_l1_loss
    return lap_loss(x, y) + 0.2 * l1_loss(x, y)
    
def evaluate(model, val_loader, loss_func=F.smooth_l1_loss):
    with torch.no_grad():
        model.eval()
        outputs = [model.validation_step(batch, acc_func=accuracy, loss_func=loss_func) for batch in val_loader]
        return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, loss_func=F.smooth_l1_loss, id=0):
    print('Starting training')
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print(f'Running epoch {epoch} ... ', end='\r')
        # Temporarily save the trained model
        with open('tmp_model.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
            
        # Training Phase 
        model.train()
        train_losses = []
        for i, batch in enumerate(train_loader):
            print(f'Running epoch {epoch} ... {i/len(train_loader)*100:3.0f}%', end='\r')
            loss = model.training_step(batch, loss_func=loss_func)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(f'Running epoch {epoch} ... Done                     ', end='\r')
        # Validation phase
        result = evaluate(model, val_loader, loss_func=loss_func)
        
        if result['val_acc'] < 0.01 or math.isnan(result['val_acc']):
            print(f"Training anomaly detected (val_acc={result['val_acc']}), aborting...")
            model.load_state_dict(torch.load('tmp_model.pt', map_location=get_default_device(id)))
            break
        
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    
def accuracy(outputs, labels):
    return pytorch_ssim.ssim(outputs, labels)
    
# If executed as script
if __name__ == '__main__':
    # Check usage
    if len(sys.argv) < 6:
        print("Usage:\n\t>>python train_nn features_path labels_path out_folder batch_size epochs gpu_id (use_pretrained)\n")
        exit()      
    # Process input parameters
    features_path = sys.argv[1]
    labels_path = sys.argv[2]
    out_folder = sys.argv[3]
    batch_size = int(sys.argv[4])
    epochs = int(sys.argv[5])
    pretrained = False if len(sys.argv) < 7 else True
    
    if pretrained == True:
        pretrained_path = sys.argv[6]
        print(f'Using pretrained model located at {pretrained_path}.')
    
    # Check if out_folder exists, else create it
    if not os.path.exists(os.path.dirname('/'.join((out_folder, 'train_history.json')))):
        try:
            os.makedirs(os.path.dirname('/'.join((out_folder, 'train_history.json'))))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    if torch.cuda.is_available():
        print('Training on GPU.')
    else:
        print('Training on CPU.')
        
    history = None
    model = None
    structure = 'RCAN'
    for id in range(torch.cuda.device_count()):
        print(f'--- Using GPU {id} ---\n')
        print('Loading dataset ...')
        train_loader, val_loader = load_dataset(features_path, labels_path, 0.8, batch_size, id)
        print('Done.')
        
        # Get neural network model 
        if structure == 'RCAN':
            model = RCAN(n_feats=64, n_resgroups=3, n_resblocks=5, reduction=16)
        else:
            model = CUNet()
        if pretrained == True:
            model.load_state_dict(torch.load(pretrained_path, map_location=get_default_device(id)))
        # Move model to GPU
        model.to(get_default_device(id))
        # Perform training
        try:
            history = fit(epochs=epochs, lr=0.001, model=model, train_loader=train_loader, 
                                                                val_loader=val_loader, loss_func=custom_loss, id=id)
            break
        except RuntimeError:
            print(f'Not enough memory on GPU {id} !')
    if history == None:
        raise RuntimeError('Not enough GPU memory available for training.')
    
    # Save the training information
    with open('/'.join((out_folder, 'train_history.json')), 'w') as f:
        json.dump(history, f)
    
    # Save the trained model
    with open('/'.join((out_folder, 'trained_model.pt')), 'wb') as f:
        torch.save(model.state_dict(), f)