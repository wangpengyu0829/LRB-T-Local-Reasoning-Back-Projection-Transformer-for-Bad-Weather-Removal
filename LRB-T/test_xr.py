import os
import sys
import time
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

cudnn.fastest = True
cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore") 

from skimage import measure
from torch.utils.data import DataLoader

from LRBT_xr import kitty
from data_png import get_test  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--haze_path',          type=str,   default='./data/Rain200L/test/rain'  )
parser.add_argument('--clean_path',         type=str,   default='./data/Rain200L/test/gt'    )
parser.add_argument('--netG',               type=str,   default='./model/pre.pth'            )
parser.add_argument('--workers',            type=int,   default=0                            )
parser.add_argument('--batchsize',          type=int,   default=1                            )
parser.add_argument('--exp',                type=str,   default='./model/result/'            )
opt = parser.parse_args()
        
test_dataset = get_test(opt.clean_path, opt.haze_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

image_name_list = glob.glob(os.path.join(opt.clean_path, '*.png'))
criterion = nn.MSELoss(size_average=True).cuda()

netG = kitty().cuda()

#netG = nn.DataParallel(kitty())
#netG = netG.cuda()

netG.load_state_dict(torch.load(opt.netG))
#netG = torch.load(opt.netG)
netG.eval()

directory = opt.exp
if not os.path.exists(directory):
    os.makedirs(directory)
    
trainLogger = open('%s/result.log' % opt.exp, 'w')

print('Model test')
psnrs = []
with torch.no_grad():
    
    start_time = time.time()
    
    for i, data_test in enumerate(test_loader, 0):
        
        key = image_name_list[i].split(opt.clean_path)[-1].split('\\')[-1]
        print(key)
        
        test_rain, test_clean = data_test
        test_clean, test_rain = test_clean.cuda(), test_rain.cuda()
        test_derain = netG(test_rain)
        
        out_img = test_derain.data
        vutils.save_image(out_img, directory+key, normalize=False, scale_each=False)

        clean_image  = test_clean.view(test_clean.shape[1], test_clean.shape[2], test_clean.shape[3]).cpu().numpy().astype(np.float32)
        derain_image = test_derain.view(test_derain.shape[1], test_derain.shape[2], test_derain.shape[3]).cpu().numpy().astype(np.float32)
        clean_image  = np.transpose(clean_image, (1,2,0))
        derain_image = np.transpose(derain_image, (1,2,0))
            
        psnr = measure.compare_psnr(clean_image, derain_image, data_range=1) 
        psnrs.append(psnr)

        print('| PSNR: %f |' % (psnr))
        
total_time = time.time() - start_time
psnr_avg = np.mean(psnrs)
print('Eval Result: | PSNR: %f |' % (psnr_avg))
print('Eval Result:  Avg-Time: {:.6f}  \n'.format((total_time/len(image_name_list)))) 

sys.stdout.flush()
trainLogger.write('Eval Result: | PSNR: %f |\n' % (psnr_avg))
trainLogger.write('Eval Result: Avg-Time: {:.6f}'.format((total_time/len(image_name_list))) )
trainLogger.flush()

