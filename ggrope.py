# @title GoldenGateRoPE2d
import math
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GoldenGateRoPE2d(nn.Module): # jerryxio.ng/posts/nd-rope
    def __init__(self, image_size, n_heads, n_freqs):
        super().__init__()
        n_zero_freqs = 0 # 8/32
        min_freq, max_freq = .8,10#.2, 20
        intv = math.pi * (math.sqrt(5)-1)/2 # mod pi instead of 2pi # pi*(sqrt5+-1)/2 ; + and - are equivalent bec mod pi
        # intv = math.pi * (math.sqrt(5)-1) # https://en.wikipedia.org/wiki/Golden_angle
        speed = torch.cat([torch.zeros(n_zero_freqs), min_freq * (max_freq/min_freq) ** torch.linspace(0,1,n_freqs-n_zero_freqs)]).unsqueeze(-1) # [n_freqs,1] # og
        angle = torch.arange(n_heads*n_freqs).reshape(n_heads, n_freqs) * intv # [n_heads, n_freqs]
        direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) # [n_heads, n_freqs, 2]
        h, w = image_size
        xlim, ylim = w/h, h/w
        y, x = torch.meshgrid(torch.linspace(-ylim, ylim, h), torch.linspace(-xlim, xlim, w), indexing="ij") # [h,w], y:row_num, x:col_num
        pos = torch.stack([x, y], dim=-1).reshape(-1,1,1,2) # [h*w,1,1,2] cartesian coords
        theta = (speed*direction*pos).sum(dim=-1) # [t,n_heads,n_freqs,2]->[t,n_heads,d_head]
        self.theta = theta
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.affine = torch.stack([cos, -sin, sin, cos], dim=-1).transpose(0,1).reshape(1,n_heads,h*w,n_freqs,2,2).to(device) # [t,n_heads,n_freqs,4]->[1,n_heads,t,n_freqs,2,2]

    def forward(self, x): # [b,h,t,d]
        return (self.affine @ x.unflatten(-1, (-1,2)).unsqueeze(-1)).flatten(-3) # @ [b,h,t,d_head//2,2,1]

# /2 better
# speed [n_freqs,1] # og best
# w/h best
# rope < ggrope < learned

# image_size=(8,8)
image_size=(20,30)
# image_size=(90,120)
n_heads=4
n_freqs=6
ggrope = GoldenGateRoPE2d(image_size, n_heads, n_freqs)

# x = torch.rand(2, *image_size, n_heads, n_freqs*2)
x = torch.rand(2, n_heads, image_size[0]*image_size[1], n_freqs*2)
out = ggrope(x)
print(out.shape)
# # print(out[0])
# theta = ggrope.theta.flatten(-2).permute(2,0,1).unsqueeze(1) # [t,n_heads,d_head][b,1,h,w]
theta = ggrope.theta.flatten(-2).T.reshape(n_heads*n_freqs, 1, *image_size) # [t,n_heads,d_head]->[d,1,h,w]
cy, cx = image_size[0]//2, image_size[1]//2
sim = torch.cos(theta-theta[...,cy,cx][...,None,None]) # [b,1,h,w]
# sim = sim.unflatten(0, (n_heads, n_freqs)).mean(1)
# print(sim.shape)

import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    npimg = img.numpy()
    print(npimg.shape)
    plt.rcParams["figure.figsize"] = (8,8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# bhwc

import torchvision
imshow(torchvision.utils.make_grid(sim, nrow=n_freqs))

