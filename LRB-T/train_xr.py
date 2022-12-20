import os
import sys
import time
import torch
import numpy
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.fastest = True
cudnn.benchmark = True

from skimage import measure
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from LRBT_xr import kitty
from data_png import get_train, get_test

import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--haze_path',         type=str,   default='./data/Rain200LS/train2/rain'  )
parser.add_argument('--clean_path',        type=str,   default='./data/Rain200LS/train2/gt'    )
parser.add_argument('--thaze_path',        type=str,   default='./data/Rain200LS/val2/rain'    )
parser.add_argument('--tclean_path',       type=str,   default='./data/Rain200LS/val2/gt'      )
parser.add_argument('--net',               type=str,   default=''                  )
parser.add_argument('--lr',                type=float, default=0.000250            )
parser.add_argument('--annealStart',       type=int,   default=0                   )
parser.add_argument('--epochs',            type=int,   default=250                 )
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=4                   )
parser.add_argument('--test_BN',           type=int,   default=1                   )
parser.add_argument('--exp',               type=str,   default='LRBT'              ) 
parser.add_argument('--display',           type=int,   default=100                 )
opt = parser.parse_args()

#opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed) 

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True
              
create_exp_dir(opt.exp)

train_dataset = get_train(opt.clean_path, opt.haze_path, 256)
test_dataset  = get_test(opt.tclean_path, opt.thaze_path)

train_loader = DataLoader(dataset=train_dataset, batch_size=opt.BN, shuffle=False, num_workers=opt.workers, drop_last=True, pin_memory=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=opt.test_BN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

trainLogger = open('%s/train.log' % opt.exp, 'w')

netG = kitty().cuda()

if opt.net != '':
    netG.load_state_dict(torch.load(opt.net))
  
netG.train()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

criterionMAE = CharbonnierLoss().cuda()
#criterionMAE = nn.SmoothL1Loss().cuda()
#criterionMAE = nn.MSELoss().cuda()
#criterionMAE = nn.L1Loss().cuda()

optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (0.9, 0.999))
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizerG, opt.epochs-3, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizerG, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)

print('Total-parameter-NetG: %s ' % (get_parameter_number(netG)) )
best_epoch = {'epoch':0, 'psnr':0}

for epoch in range(opt.epochs):
    print()
    Loss_img = 0.0
    ganIterations = 0
    start_time = time.time()
    print("Epoch: %d Learning Rate: %f" % (epoch+1, optimizerG.param_groups[0]['lr']))
        
    for i, data_train in enumerate(train_loader):

        rain, clean = data_train  
        clean, rain = clean.cuda(), rain.cuda()
        
        derain = netG(rain)
        netG.zero_grad()
    
        img_loss = criterionMAE(derain, clean) 
        
        img_loss.backward()  
        Loss_img += img_loss.item()
        
        optimizerG.step()
        ganIterations += 1
        
        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d] img_loss: %f' % (epoch+1, opt.epochs, i+1, len(train_loader), Loss_img*opt.BN) )
            sys.stdout.flush()
            trainLogger.write('[%d/%d][%d/%d] img_loss: %f\n' % (epoch+1, opt.epochs, i+1, len(train_loader), Loss_img*opt.BN) )
            trainLogger.flush()
            Loss_img = 0.0  
    scheduler.step() 
    
    print('Model eval')
    netG.eval()
    psnrs = []
    ssims = []
    with torch.no_grad():
        for j, data_test in enumerate(test_loader):
            
            test_rain, test_clean = data_test
            test_clean, test_rain = test_clean.cuda(), test_rain.cuda()
            
            test_derain = netG(test_rain)
            
            clean_image  = test_clean.view(test_clean.shape[1], test_clean.shape[2], test_clean.shape[3]).cpu().numpy().astype(np.float32)
            derain_image = test_derain.view(test_derain.shape[1], test_derain.shape[2], test_derain.shape[3]).cpu().numpy().astype(np.float32)
            clean_image  = np.transpose(clean_image, (1,2,0))
            derain_image = np.transpose(derain_image, (1,2,0))
               
            psnr = measure.compare_psnr(clean_image, derain_image, data_range=1)   
            psnrs.append(psnr)                                                     
            ssim = measure.compare_ssim(clean_image, derain_image, data_range=1, multichannel=True)
            ssims.append(ssim) 

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims) 
    print('Eval Result: [%d/%d] | PSNR: %f | SSIM: %f' % (epoch+1, opt.epochs, psnr_avg, ssim_avg))
    sys.stdout.flush()
    trainLogger.write('Eval Result: [%d/%d] | PSNR: %f | SSIM: %f\n' % (epoch+1, opt.epochs, psnr_avg, ssim_avg) )
    trainLogger.flush()
    netG.train()
    
    if psnr_avg > best_epoch['psnr']:
        torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (opt.exp, epoch+1))
        best_epoch['psnr'] = psnr_avg
        best_epoch['epoch'] = epoch+1 

    total_time = time.time() - start_time
    print('Total-Time: {:.6f} '.format(total_time))  
    
trainLogger.close()


