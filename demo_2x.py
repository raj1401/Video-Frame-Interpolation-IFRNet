import os
import numpy as np
import torch
from models.IFRNet import Model
from utils import read
from imageio import mimsave

from timing import timing_wrapper

@timing_wrapper
def time_minference(img1, img2, embt):
    imgt_pred = model.inference(img1, img2, embt)

cwd = os.getcwd()
path = os.path.join(cwd, 'checkpoints/IFRNet/IFRNet_Vimeo90K.pth')

model = Model().cuda().eval()
model.load_state_dict(torch.load(path))

img0_path = os.path.join(cwd, 'figures/img0.png') 
img0_np = read(img0_path)

img1_path = os.path.join(cwd, 'figures/img1.png') 
img1_np = read(img1_path)

img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

time_minference(img0, img1, embt)
imgt_pred = model.inference(img0, img1, embt)


imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

images = [img0_np, imgt_pred_np, img1_np]
gif_path = os.path.join(cwd, 'figures/out_2x.gif')
mimsave(gif_path, images, fps=3)
