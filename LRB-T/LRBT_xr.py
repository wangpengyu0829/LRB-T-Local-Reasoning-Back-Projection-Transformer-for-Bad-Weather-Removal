
import torch
import torch.nn as nn
from involution_xr import involution
from timm.models.layers import DropPath 

class MSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
   
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.zpool   = ZPool()
        self.conv    = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        x1 = self.zpool(x)
        x2 = self.conv(x1)
        weight = self.sigmoid(x2) 
        return weight     

class Inv_block(nn.Module):
    def __init__(self, channel):
        super().__init__() 
        self.conv1 = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.inv   = involution(channels=channel//2, ksize=7, stride=1, group_ch=16, red_ratio=2)
        self.relu2 = nn.PReLU()
        self.conv2 = nn.Conv2d(channel//2, channel, kernel_size=1, stride=1, padding=0)
        self.attn = Attn()
        self.psec = nn.Sequential(
                    nn.Conv2d(channel//4, channel//4, kernel_size=3, stride=1, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(channel//4, channel, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        f, x2 = self.inv(x1)
        x3 = self.conv2(self.relu2(x2))
        x4 = x3 * self.attn(f)
        x5 = x4 + self.psec(f)
        return x5   
    
class INV_down(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.inv  = Inv_block(hidden_features)
        self.down = nn.Unfold(kernel_size=2, stride=2, padding=0)
        self.fc2  = nn.Linear(hidden_features*4, in_features*4)
        self.drop = nn.Dropout(drop)

    def forward(self, x, y):
        _, _, c = x.size()
        b, _, h, w = y.size()
        t2i = self.fc1(x).transpose(1, 2).reshape(b, c, h, w)
        mid = self.inv(t2i) + t2i
        i2t = self.fc2(self.down(mid).transpose(1, 2))
        x = self.drop(i2t)
        return x   
    
class INV_up(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.inv  = Inv_block(hidden_features) 
        self.up   = nn.Sequential(
                    nn.PixelShuffle(2),
                    nn.Unfold(kernel_size=1, stride=1, padding=0))
        self.fc2  = nn.Linear(hidden_features//4, in_features//4)
        self.drop = nn.Dropout(drop)

    def forward(self, x, y):
        _, _, c = x.size()
        b, _, h, w = y.size()
        t2i = self.fc1(x).transpose(1, 2).reshape(b, c, h, w)
        mid = self.inv(t2i) + t2i
        i2t = self.fc2(self.up(mid).transpose(1, 2))
        x = self.drop(i2t)
        return x
        
class Up_transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.MSA = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.MLP = INV_up(in_features=dim, hidden_features=dim*mlp_ratio, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.MSA(self.norm1(x)))
        x = self.drop_path(self.MLP(self.norm2(x), y))
        return x
    
class Down_transformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, attn_drop=0.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.MSA = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.MLP = INV_down(in_features=dim, hidden_features=dim*mlp_ratio, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.MSA(self.norm1(x)))
        x = self.drop_path(self.MLP(self.norm2(x), y))
        return x
    
class Up_block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()   
        self.up   = nn.Upsample(scale_factor=2.0)
        self.up1 = Up_transformer(dim=dim, num_heads=num_heads, mlp_ratio=1, drop=0.0, attn_drop=0.0)
        self.down2 = Down_transformer(dim=dim//4, num_heads=num_heads, mlp_ratio=1, drop=0.0, attn_drop=0.0)
        self.up3 = Up_transformer(dim=dim, num_heads=num_heads, mlp_ratio=1, drop=0.0, attn_drop=0.0)

    def forward(self, token, image):
        token0 = token
        image_large = self.up(image)
        image_small = image
        token1 = self.up1(token0, image_small)
        token2 = self.down2(token1, image_large) - token0
        token3 = self.up3(token2, image_small) + token1
        return token3

class Down_block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.down = nn.Upsample(scale_factor=0.5)
        self.down1 = Down_transformer(dim=dim, num_heads=num_heads, mlp_ratio=1, drop=0.0, attn_drop=0.0)
        self.up2 = Up_transformer(dim=dim*4, num_heads=num_heads, mlp_ratio=1, drop=0.0, attn_drop=0.0)
        self.down3 = Down_transformer(dim=dim, num_heads=num_heads, mlp_ratio=1, drop=0.0, attn_drop=0.0)

    def forward(self, token, image):
        token0 = token
        image_large = image
        image_small = self.down(image)
        token1 = self.down1(token0, image_large)
        token2 = self.up2(token1, image_small) - token0
        token3 = self.down3(token2, image_large) + token1
        return token3

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class Res_block(nn.Module):
    def __init__(self, channel):
        super(Res_block, self).__init__() 
        self.demo = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out = self.demo(x).mul_(0.1)
        return out

class RFA(nn.Module):
    def __init__(self, channel):
        super(RFA, self).__init__()
        self.RB1  = Res_block(channel)
        self.RB2  = Res_block(channel)
        self.RB3  = Res_block(channel)
        self.RB4  = Res_block(channel)
        self.conv = nn.Conv2d(channel*4, channel, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x0 = x
        y1 = self.RB1(x0)
        x1 = y1 + x0
        y2 = self.RB2(x1)
        x2 = y2 + x1
        y3 = self.RB3(x2)
        x3 = y3 + x2
        y4 = self.RB4(x3)
        y = torch.cat([y1, y2, y3, y4], 1)
        out = self.conv(y) + x
        return out

class Up_uint(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up_uint, self).__init__() 
        self.demo = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch*4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2) ) 
        
    def forward(self, x):
        out = self.demo(x)
        return out

class Down_uint(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_uint, self).__init__()
        self.demo = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch,  kernel_size=3, stride=2, padding=1, bias=False) )

    def forward(self, x):
        out = self.demo(x)
        return out


class kitty(nn.Module):
    def __init__(self, num_heads=2, ch=32):
        super().__init__()
        
        self.down4x  = nn.Conv2d(ch*2, ch*2,  kernel_size=3, stride=2, padding=1)
        self.down8x  = nn.Conv2d(ch*2, ch*2,  kernel_size=3, stride=2, padding=1)
        self.down16x = nn.Conv2d(ch*2, ch*2,  kernel_size=3, stride=2, padding=1)
        
        self.demo4x  = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(ch*2, ch, kernel_size=1, stride=1, padding=0),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0), 
                       nn.ReLU(inplace=True) )
        
        self.demo8x  = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(ch*2, ch, kernel_size=1, stride=1, padding=0),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0), 
                       nn.ReLU(inplace=True) )
        
        self.demo16x = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(ch*2, ch, kernel_size=1, stride=1, padding=0),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0), 
                       nn.ReLU(inplace=True) )
        
        self.conv = ConvLayer(3, ch, kernel_size=7, stride=1)
        self.uint11 = RFA(ch)
        self.uint12 = RFA(ch)
        self.uint13 = RFA(ch)
        self.down1 = Down_uint(ch, ch*2)
        
        self.uint21 = RFA(ch*2)
        self.uint22 = RFA(ch*2)
        self.down2 = nn.Unfold(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(ch*8, ch*2)
        
        self.stage1  = Down_block(dim=ch*2, num_heads=num_heads)
        self.linear1 = nn.Linear(ch*8, ch*4)

        self.stage2  = Down_block(dim=ch*4, num_heads=num_heads)
        self.linear2 = nn.Linear(ch*16, ch*8)
        
        self.stage3_1 = Up_block(dim=ch*8, num_heads=num_heads)
        self.stage3_2 = Down_block(dim=ch*2, num_heads=num_heads)
        self.stage3_3 = Up_block(dim=ch*8, num_heads=num_heads)
        self.stage3_4 = Down_block(dim=ch*2, num_heads=num_heads)
        
        self.stage4 = Up_block(dim=ch*16, num_heads=num_heads)
        
        self.stage5 = Up_block(dim=ch*8, num_heads=num_heads)
        
        self.fc2 = nn.Linear(ch*4, ch*4)
        
        self.up1 = Up_uint(ch*4, ch*2)
        self.conv3 = nn.Conv2d(ch*4, ch*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.uint31 = RFA(ch*2)
        self.uint32 = RFA(ch*2)

        self.up2 = Up_uint(ch*2, ch)
        self.conv4 = nn.Conv2d(ch*2, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.uint41 = RFA(ch)
        self.uint42 = RFA(ch)
        self.uint43 = RFA(ch)
        
        self.fine = ConvLayer(ch, 3, kernel_size=3, stride=1)

    def forward(self, x):
  
        img1 = self.conv(x)
        img1 = self.uint13(self.uint12(self.uint11(img1)))
        img2 = self.down1(img1) 
        img2 = self.uint22(self.uint21(img2))
        img = self.down2(img2) 

        img_4x  = self.down4x(img2)
        par_4x  = self.demo4x(img_4x).squeeze(dim=3) + 1.0
        b4x, c4x, h4x, w4x = img_4x.size()
        
        img_8x  = self.down8x(img_4x)
        par_8x  = self.demo8x(img_8x).squeeze(dim=3) + 1.0
        b8x, c8x, h8x, w8x = img_8x.size()
        
        img_16x = self.down16x(img_8x)
        par_16x = self.demo16x(img_16x).squeeze(dim=3) + 1.0
        b16x, c16x, h16x, w16x = img_16x.size()
        
        tkn = self.fc1(img.transpose(1, 2))
        tkn1 = self.stage1(tkn, img_4x)
        tkn1 = self.linear1(tkn1)
        
        tkn2 = self.stage2(tkn1, img_8x)
        tkn2 = self.linear2(tkn2)
        
        tkn3_1 = self.stage3_1(tkn2,  img_16x)
        tkn3_2 = self.stage3_2(tkn3_1, img_8x)
        tkn3_3 = self.stage3_3(tkn3_2, img_16x)
        tkn3_4 = self.stage3_4(tkn3_3, img_8x)
        tkn3 = tkn3_4
        
        tkn4 = self.stage4(torch.cat([tkn3, par_16x*tkn2],2), img_16x)
        tkn5 = self.stage5(torch.cat([tkn4, par_8x*tkn1],2), img_8x) 
        
        t2i = self.fc2(torch.cat([tkn5, par_4x*tkn],2))
        t2i = t2i.transpose(1, 2).reshape(b4x, -1, h4x, w4x)

        img3 = self.conv3(torch.cat([self.up1(t2i), img2],1))
        img3 = self.uint32(self.uint31(img3))

        img4 = self.conv4(torch.cat([self.up2(img3), img1],1))
        img4 = self.uint43(self.uint42(self.uint41(img4)))

        out = self.fine(img4)
        out = x - out
        return out
