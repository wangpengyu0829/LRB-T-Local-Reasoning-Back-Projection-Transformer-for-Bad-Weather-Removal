import os
import cv2
#import math
import torch
import random
import numpy as np
import torch.utils.data as data

def is_img1(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False
    
def is_img2(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False
    
def _np2Tensor(img):  
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float() 
    return tensor

class get_train(data.Dataset):
    def __init__(self, clean_path, haze_path, patch_size):
        self.patch_size = patch_size   
        self.haze_path  = haze_path 
        self.clean_path = clean_path 
        self._set_filesystem(self.haze_path, self.clean_path)   
        self.images_h, self.images_c = self._scan()             
        self.repeat = 2
               
    def _set_filesystem(self, dir_h, dir_c):
        self.dir_h = dir_h
        self.dir_c = dir_c
        print('********* Train dir *********')
        print(self.dir_h)
        print(self.dir_c)
        
    def _scan(self):
        list_c = sorted([os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c) if is_img1(x)])  
        random.shuffle(list_c)
        list_h = [os.path.splitext(x)[0]+'.png' for x in list_c]
        list_h = [os.path.join(self.dir_h, os.path.split(x)[-1]) for x in list_h]  
        return list_h, list_c                  

    def __getitem__(self, idx):
        img_h, img_c, filename_h, filename_c = self._load_file(idx)     
        assert img_h.shape==img_c.shape 
        x = random.randint(0, img_h.shape[0] - self.patch_size)          
        y = random.randint(0, img_h.shape[1] - self.patch_size)
        img_h = img_h[x : x+self.patch_size, y : y+self.patch_size, :]   
        img_c = img_c[x : x+self.patch_size, y : y+self.patch_size, :]
        img_h = _np2Tensor(img_h)                                        
        img_c = _np2Tensor(img_c)
        return img_h, img_c

    def __len__(self):
        return len(self.images_h) * self.repeat
        
    def _get_index(self, idx):
        return idx % len(self.images_h)  

    def _load_file(self, idx):
        idx = self._get_index(idx)    
        file_h = self.images_h[idx] 
        file_c = self.images_c[idx]   
        img_h = cv2.cvtColor(cv2.imread(file_h),   cv2.COLOR_BGR2RGB)  
        if np.max(img_h)>1: img_h = img_h/255.0 
#        if img_h.shape[0]<256: img_h = cv2.resize(img_h, (img_h.shape[1], 256), interpolation=cv2.INTER_CUBIC)
#        if img_h.shape[1]<256: img_h = cv2.resize(img_h, (256, img_h.shape[0]), interpolation=cv2.INTER_CUBIC)      
        
        img_c = cv2.cvtColor(cv2.imread(file_c),   cv2.COLOR_BGR2RGB)
        if np.max(img_c)>1: img_c = img_c/255.0    
#        if img_c.shape[0]<256: img_c = cv2.resize(img_c, (img_c.shape[1], 256), interpolation=cv2.INTER_CUBIC)
#        if img_c.shape[1]<256: img_c = cv2.resize(img_c, (256, img_c.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        filename_h = os.path.splitext(os.path.split(file_h)[-1])[0]   
        filename_c = os.path.splitext(os.path.split(file_c)[-1])[0]   
        return img_h, img_c, filename_h, filename_c                  
    
    
class get_test(data.Dataset):
    def __init__(self, clean_path, haze_path):
        self.haze_path  = haze_path 
        self.clean_path = clean_path 
        self._set_filesystem(self.haze_path, self.clean_path)  
        self.images_h, self.images_c = self._scan()             
                
    def _set_filesystem(self, dir_h, dir_c):
        self.dir_h = dir_h
        self.dir_c = dir_c
        print('********* Test dir *********')
        print(self.dir_h)
        print(self.dir_c)
        
    def _scan(self):
        list_c = sorted([os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c) if is_img2(x)])
        list_h = [os.path.splitext(x)[0]+'.png' for x in list_c]
        list_h = [os.path.join(self.dir_h, os.path.split(x)[-1]) for x in list_h]  
        return list_h, list_c                  

    def __getitem__(self, idx):
        img_h, img_c, filename_h, filename_c = self._load_file(idx) 
        assert img_h.shape==img_c.shape 
        img_h = _np2Tensor(img_h)      
        img_c = _np2Tensor(img_c)
        return img_h, img_c

    def __len__(self):
        return len(self.images_h)
        
    def _get_index(self, idx):
        return idx  

    def _load_file(self, idx):
        idx = self._get_index(idx)    
        file_h = self.images_h[idx]   
        file_c = self.images_c[idx]  
        
        img_h = cv2.cvtColor(cv2.imread(file_h), cv2.COLOR_BGR2RGB) 
        if np.max(img_h)>1: img_h = img_h/255.0 
        
#        xh = math.floor(img_h.shape[0]/16)*16
#        yh = math.floor(img_h.shape[1]/16)*16
#        img_h = cv2.resize(img_h, (yh, xh), interpolation=cv2.INTER_CUBIC)
        
        img_c = cv2.cvtColor(cv2.imread(file_c), cv2.COLOR_BGR2RGB)
        if np.max(img_c)>1: img_c = img_c/255.0   # 归一化
        
#        xc = math.floor(img_c.shape[0]/16)*16
#        yc = math.floor(img_c.shape[1]/16)*16
#        img_c = cv2.resize(img_c, (yc, xc), interpolation=cv2.INTER_CUBIC)   
        
        filename_h = os.path.splitext(os.path.split(file_h)[-1])[0]   
        filename_c = os.path.splitext(os.path.split(file_c)[-1])[0] 
        return img_h, img_c, filename_h, filename_c                   


